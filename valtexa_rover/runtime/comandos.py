"""Comandos do centro de comando: o robô busca, confere a assinatura, executa e responde.

O canal é o mesmo protocolo assinado da missão (``saas`` ``/v1/device/commands``): cada
envelope traz destinatário, sequência, nonce e validade, conferidos por
:func:`~valtexa_rover.runtime.mission.verify_command` com uma sequência **própria** — um
comando remoto nunca é recusado por causa de uma missão, nem o contrário.

Tipos: ``diagnostico`` (laudo de cada peça, com foto reduzida de cada câmera se pedido),
``foto`` (uma câmera), ``modo`` (autonomo | manual | pausado | parado, para a navegação),
``parar``, ``config`` (troca parte do ``rover.json`` — validada, troca atômica com
backup, vale ao reiniciar o serviço, como no OTA), ``recarregar_modelos`` (modelo fixado
pela equipe), ``reiniciar``, ``ensaio``, ``ensaio_cancelar``, ``liberar_retidos`` (volta à
fila o que o orçamento de dados reteve no disco: ``{"motivo"?, "record_ids"?, "maximo"?}``)
e ``orcamento`` (o consumo de dados do dia, com o que espera e o que está retido).

**Ensaio.** ``ensaio`` (``{"ensaio": "gnss", "params": {...}}``) roda numa thread
(:mod:`valtexa_rover.ensaios`): o comando fica "em execução" na plataforma, o progresso
vai por ``POST /v1/device/commands/<id>/progresso`` e o laudo, no fim, pelo resultado de
sempre. Sem rede no fim, o laudo fica guardado e vai na próxima busca. Só um por vez, e
nunca com missão em curso. ``ensaio_cancelar`` (``{"command_id": ...}``) para o ensaio no
próximo ponto de checagem. Ensaio não move o robô.

**Controle remoto.** Em modo ``manual``, a cada ciclo o Runtime busca o ponto de ajuste
assinado (``/v1/device/teleop``) e o passa à navegação (``teleop.json``). Sem ponto
válido — a equipe soltou o comando, caiu a rede, venceu o 1 s —, escreve zero: a
navegação para. O Runtime não comanda motor; quem obedece ao ``teleop.json`` é a pilha de
navegação/segurança do Rover, que também tem de parar sozinha quando ``valid_until`` passa.
O Runtime também escreve zero quando o ponto que está no ``teleop.json`` vence sem outro
(o servidor repetindo o mesmo ponto): o arquivo nunca fica com velocidade vencida.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from valtexa_rover import ensaios
from valtexa_rover.config import ConfigError, RuntimeConfig
from valtexa_rover.hal import diagnostico as diag
from valtexa_rover.hal.interfaces import NetworkUnavailableError
from valtexa_rover.runtime.mission import CommandRejectedError, verify_command

SEQ_COMANDO = "last_remote_command_sequence"
SEQ_TELEOP = "last_teleop_sequence"


def _mescla(base: dict, mudancas: dict) -> dict:
    saida = dict(base)
    for chave, valor in mudancas.items():
        if isinstance(valor, dict) and isinstance(saida.get(chave), dict):
            saida[chave] = _mescla(saida[chave], valor)
        else:
            saida[chave] = valor
    return saida


class CommandExecutor:
    def __init__(self, core, config_path: Path | str | None = None) -> None:
        self.core = core
        self.hardware = core.hardware
        self.config_path = Path(config_path) if config_path else core.store.root / "rover.json"
        self.restart_requested = False
        self.video = None
        """:class:`~valtexa_rover.runtime.video.TransmissorDeVideo`, se o serviço ligou."""
        self._ultimo_teleop: int | None = None
        self._teleop_ate: datetime | None = None  # valid_until do ponto que está no arquivo
        self._laudos_pendentes: list[tuple[str, dict]] = []
        self._lock_laudos = threading.Lock()
        self.ensaios = ensaios.ExecutorDeEnsaios(
            core.hardware, core.clock, lambda: ensaios.fontes_do_runtime(core),
            ao_progresso=self._progresso_ensaio, ao_terminar=self._fim_ensaio)

    # -- comandos ---------------------------------------------------------------

    def poll(self) -> int:
        """Busca, confere, executa e responde. Devolve quantos executou."""
        identidade = self.core.identity
        if not identidade.platform_public_key:
            return 0
        self._envia_laudos_pendentes()
        try:
            resposta = self.core.client.request("GET", "/v1/device/commands")
        except NetworkUnavailableError:
            return 0
        if resposta.status != 200:
            return 0
        if self.video is not None:  # alguém assiste? o pedido vem nesta mesma resposta
            cab = {k.lower(): v for k, v in (resposta.headers or {}).items()}
            self.video.atualiza(cab.get("x-valtexa-video"))
        envelopes = sorted(resposta.json() or [],
                           key=lambda e: int((e.get("payload") or {}).get("sequence", 0)))
        feitos = 0
        for envelope in envelopes:
            try:
                corpo = verify_command(
                    envelope, device_id=identidade.device_id,
                    platform_public_key=identidade.platform_public_key, store=self.core.store,
                    now=self.core.clock.now(), sequence_key=SEQ_COMANDO,
                    command_prefix="device.", piso=self._piso())
            except CommandRejectedError as exc:
                self.core.log.warning("comando remoto recusado",
                                      extra={"data": {"motivo": str(exc)}})
                continue
            if corpo.get("kind") == "ensaio":
                self._inicia_ensaio(corpo["command_id"], corpo.get("params") or {})
                feitos += 1
                continue
            ok, resultado, erro = self.execute(corpo.get("kind", ""), corpo.get("params") or {})
            self.core.log.info("comando remoto", extra={"data": {
                "kind": corpo.get("kind"), "ok": ok, "erro": erro}})
            # Executado; sem rede, o painel mostra "entregue" até a rede voltar.
            with contextlib.suppress(NetworkUnavailableError):
                self.core.client.request(
                    "POST", f"/v1/device/commands/{corpo['command_id']}/result",
                    json_body={"ok": ok, "result": resultado, "error": erro})
            feitos += 1
        return feitos

    def execute(self, kind: str, params: dict) -> tuple[bool, dict | None, str | None]:
        try:
            metodo = getattr(self, f"_cmd_{kind}", None)
            if metodo is None:
                return False, None, f"comando desconhecido: {kind!r}"
            return True, metodo(params), None
        except Exception as exc:  # o comando falha; o Runtime segue
            return False, None, f"{type(exc).__name__}: {exc}"

    def _cmd_diagnostico(self, params: dict) -> dict:
        return diag.run(self.hardware, params.get("pecas"), fotos=bool(params.get("fotos")),
                        espera_gnss=float(params.get("espera_gnss", 3.0)))

    def _cmd_foto(self, params: dict) -> dict:
        alvo = params["camera_id"]
        camera = next((c for c in self.hardware.cameras if c.camera_id == alvo), None)
        if camera is None:
            raise LookupError(f"câmera {alvo!r} não existe neste rover")
        q = camera.capture()
        return {"camera_id": alvo, "px": [q.width, q.height], "distancia_mm": q.distance_mm,
                "altura_mm": q.camera_height_mm, "hfov_deg": q.hfov_deg,
                "captured_at": q.captured_at.isoformat(),
                **diag.thumbnail(q.data, lado=int(params.get("lado", 960)))}

    def _controle(self):
        controle = self.hardware.controller
        if controle is None or not hasattr(controle, "set_mode"):
            raise RuntimeError("rover sem navegação configurada (hardware.json: navegacao)")
        return controle

    def _cmd_modo(self, params: dict) -> dict:
        self._controle().set_mode(params["modo"])
        if params["modo"] != "manual":
            self._zera_teleop()
        return {"modo": params["modo"]}

    def _cmd_parar(self, params: dict) -> dict:
        controle = self._controle()
        controle.set_mode("parado")
        # Depois do modo: o motivo de quem mandou parar é o que fica no parar.json.
        controle.stop(params.get("motivo") or "parada pelo centro de comando")
        self._zera_teleop()
        return {"modo": "parado"}

    def _cmd_config(self, params: dict) -> dict:
        if self.ensaios.ocupado is not None:
            raise RuntimeError("ensaio em curso: cancele antes de trocar a configuração")
        atual = (json.loads(self.config_path.read_text(encoding="utf-8"))
                 if self.config_path.exists() else RuntimeConfig().to_dict())
        nova = _mescla(atual, params["rover_json"])
        try:
            RuntimeConfig.from_dict(nova).validate()
        except ConfigError as exc:
            raise ValueError(f"configuração inválida: {exc}") from exc
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        if self.config_path.exists():
            shutil.copyfile(self.config_path, self.config_path.with_suffix(".json.anterior"))
        temp = self.config_path.with_suffix(".json.part")
        temp.write_text(json.dumps(nova, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temp, self.config_path)
        self.restart_requested = True  # como no OTA: vale ao reiniciar o serviço
        return {"aplicada": True, "reinicia": True, "secoes": sorted(params["rover_json"])}

    def _cmd_recarregar_modelos(self, params: dict) -> dict:
        self.core.models.refresh()
        missao = self.core.missions.current()
        if missao is not None and missao.field_id:
            self.core.models.activate_for_field(missao.field_id, missao.id)
        else:
            motivo = self.core.precarrega_modelo()
            if motivo:
                raise RuntimeError(motivo)
        return {"active_model": self.core.store.get("active_model")}

    def _cmd_liberar_retidos(self, params: dict) -> dict:
        from valtexa_rover.runtime.orcamento import MOTIVOS

        motivo = params.get("motivo")
        if motivo is not None and motivo not in MOTIVOS:
            raise ValueError(f"motivo desconhecido: {motivo!r} (use {', '.join(MOTIVOS)}).")
        ids = params.get("record_ids")
        if ids is not None and (not isinstance(ids, list)
                                or not all(isinstance(i, str) for i in ids) or len(ids) > 500):
            raise ValueError("record_ids: lista de até 500 textos.")
        maximo = int(params.get("maximo", 100))
        if not 1 <= maximo <= 5000:
            raise ValueError("maximo entre 1 e 5000.")
        n = self.core.orcamento.liberar_retidos(motivo=motivo, record_ids=ids, maximo=maximo)
        return {"liberados": n, "relatorio": self.core.orcamento.relatorio()}

    def _cmd_orcamento(self, params: dict) -> dict:
        return self.core.orcamento.relatorio(params.get("dia"))

    def _cmd_reiniciar(self, params: dict) -> dict:
        if self.ensaios.ocupado is not None:
            raise RuntimeError("ensaio em curso: cancele antes de reiniciar")
        self.restart_requested = True
        return {"reinicia": True}

    # -- ensaios ----------------------------------------------------------------------------

    def _inicia_ensaio(self, command_id: str, params: dict) -> None:
        """Começa na thread e responde só o progresso; o laudo vai em :meth:`_fim_ensaio`."""
        try:
            if self.core.missions.current() is not None:
                raise ensaios.EnsaioIndisponivelError(
                    "missão em curso: ensaio só com o robô fora de missão")
            inicio = self.ensaios.iniciar(command_id, str(params.get("ensaio", "")),
                                          params.get("params") or {})
        except (ensaios.ParametroInvalidoError, ensaios.EnsaioIndisponivelError) as exc:
            self.core.log.warning("ensaio recusado", extra={"data": {"motivo": str(exc)}})
            self._posta_resultado(command_id, {"ok": False, "result": None, "error": str(exc)})
            return
        self.core.log.info("ensaio iniciado", extra={"data": inicio})
        self._progresso_ensaio(command_id, 0.0, "iniciado", {"params": inicio["params"]})

    def _progresso_ensaio(self, command_id: str, fracao: float, etapa: str,
                          parcial: dict | None) -> None:
        with contextlib.suppress(NetworkUnavailableError):
            self.core.client.request(
                "POST", f"/v1/device/commands/{command_id}/progresso",
                json_body={"fracao": round(fracao, 4), "etapa": etapa[:200],
                           "parcial": parcial})

    def _fim_ensaio(self, command_id: str, laudo: dict) -> None:
        self.core.log.info("ensaio terminado", extra={"data": {
            "ensaio": laudo.get("ensaio"), "status": laudo.get("status")}})
        corpo = {"ok": laudo.get("status") != "erro", "result": laudo,
                 "error": laudo.get("erro")}
        if not self._posta_resultado(command_id, corpo):
            with self._lock_laudos:
                self._laudos_pendentes.append((command_id, corpo))

    def _posta_resultado(self, command_id: str, corpo: dict) -> bool:
        try:
            r = self.core.client.request("POST", f"/v1/device/commands/{command_id}/result",
                                         json_body=corpo)
        except NetworkUnavailableError:
            return False
        return r.status < 500

    def _envia_laudos_pendentes(self) -> None:
        with self._lock_laudos:
            pendentes, self._laudos_pendentes = self._laudos_pendentes, []
        for command_id, corpo in pendentes:
            if not self._posta_resultado(command_id, corpo):
                with self._lock_laudos:
                    self._laudos_pendentes.append((command_id, corpo))

    def _cmd_ensaio_cancelar(self, params: dict) -> dict:
        alvo = str(params.get("command_id") or "")
        return {"command_id": alvo, "cancelado": self.ensaios.cancelar(alvo)}

    # -- controle remoto --------------------------------------------------------

    def _piso(self):
        return getattr(self.core, "piso_plataforma", None)

    def _agora(self) -> datetime:
        """Hora para validade: o relógio do robô, nunca antes da hora assinada da plataforma."""
        piso = self._piso()
        agora = self.core.clock.now()
        return agora if piso is None else piso.agora(agora)

    def _zera_teleop(self) -> None:
        controle = self.hardware.controller
        if controle is not None and hasattr(controle, "teleop"):
            controle.teleop(0.0, 0.0, None, seq=self._ultimo_teleop or 0)
        self._teleop_ate = None

    def teleop(self) -> dict[str, Any] | None:
        """Em modo manual: o ponto assinado vai à navegação; sem ponto, zero."""
        controle = self.hardware.controller
        if controle is None or not hasattr(controle, "teleop"):
            return None
        if getattr(controle, "mode", lambda: None)() != "manual":
            return None
        identidade = self.core.identity
        try:
            resposta = self.core.client.request("GET", "/v1/device/teleop")
        except NetworkUnavailableError:
            self._zera_teleop()  # sem enlace, sem controle remoto: pare
            return None
        if resposta.status != 200 or not identidade.platform_public_key:
            self._zera_teleop()
            return None
        envelope = resposta.json()
        seq = int((envelope.get("payload") or {}).get("sequence", 0))
        if seq == self._ultimo_teleop:
            # O mesmo ponto: já está na navegação. Venceu sem outro → zero no arquivo.
            if self._teleop_ate is not None and self._agora() >= self._teleop_ate:
                self._zera_teleop()
            return None
        try:
            corpo = verify_command(
                envelope, device_id=identidade.device_id,
                platform_public_key=identidade.platform_public_key, store=self.core.store,
                now=self.core.clock.now(), sequence_key=SEQ_TELEOP,
                command_prefix="device.teleop", piso=self._piso())
        except CommandRejectedError:
            self._zera_teleop()
            return None
        self._ultimo_teleop = seq
        controle.teleop(float(corpo["v"]), float(corpo["w"]), corpo.get("valid_until"), seq=seq)
        try:
            self._teleop_ate = datetime.fromisoformat(corpo["valid_until"])
        except (KeyError, TypeError, ValueError):
            self._teleop_ate = self.core.clock.now()  # sem prazo legível: vale um ciclo
        return corpo
