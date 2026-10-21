"""Perda de enlace no Rover: avisa a navegação e aplica a política da missão.

A cada volta, :class:`PoliticaDeEnlace` escreve ``/run/valtexa/enlace.json`` (escrita
atômica) com a saúde de cada enlace e o que foi decidido::

    {"at": "...", "algum_vivo": false, "sem_enlace_desde": "...", "sem_enlace_s": 4.2,
     "melhor": null, "enlaces": {"subghz": {"estado": "morto", "rtt_ms": ..., ...}},
     "nuvem": true, "acao": "pausado", "missao_permite_sem_enlace": false}

Sem nenhum enlace vivo por mais de ``tolerancia_s``:

* **teleop zera** sempre (o ``teleop.json`` vai a zero; a navegação também para sozinha
  quando o ``valid_until`` do último ponto passa);
* com missão autônoma em curso, ela **só continua se a missão permitir**
  (``politica_enlace.continuar_sem_enlace`` no corpo assinado da missão); senão o modo
  vai a ``pausado`` — e volta a ``autonomo`` sozinho quando um enlace religar, se quem
  pausou foi esta política;
* em modo ``manual``, vai a ``pausado`` (sem operador não há quem dirija).

**Política por missão** (Parte 13, :class:`PoliticaDaMissao`, no corpo assinado):

* ``politica_enlace`` explícita manda: ``continuar_sem_enlace`` (padrão não),
  ``atravessar_zonas_mapeadas`` (padrão sim), ``travessia_max_s`` (120), ``tolerancia_s``,
  ``voltar_apos_s`` (pausado há este tempo → pede à navegação o retorno ao último ponto com
  enlace; padrão nunca) e ``fechar_aplicacao`` (padrão não);
* sem ela, vem do ``offline_policy`` que a plataforma já assina em toda missão: pulverização
  (``on_link_loss: pause`` ou ``stop_application``) **nunca segue às cegas** — pausa, fecha
  a aplicação e não atravessa zona morta; ``after_grace: return_to_base`` vira
  ``voltar_apos_s = grace_seconds``. Missão de varredura atravessa a zona mapeada até
  ``min(120, grace_seconds)`` e pausa na perda **inesperada** (como na P12);
* nenhuma das duas (missão local, testes): o padrão da P12 + travessia de zona mapeada.

**Travessia planejada** (:mod:`.travessia`): com o mapa de cobertura, a perda dentro de uma
zona morta conhecida e dentro da duração prevista (com folga) é **esperada**: a missão segue
(``acao: "travessia_planejada"``) se a política deixar; passou do previsto, ou do
``travessia_max_s``, vira perda inesperada.

**Parar nunca depende do enlace.** Esta política só *reduz* movimento quando o rádio cai;
a parada de emergência é o rádio próprio ligado ao MCU, e o ``parar.json`` local não passa
por aqui. O Runtime não comanda motor: escreve arquivos que a navegação e o MCU leem.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from valtexa_rover._compat import UTC

ACAO_NENHUMA = "nenhuma"
ACAO_TELEOP_ZERADO = "teleop_zerado"
ACAO_PAUSADO = "pausado"
ACAO_CONTINUA = "continua_autonomo"
ACAO_RETOMADO = "retomado"
ACAO_TRAVESSIA = "travessia_planejada"
ACAO_VOLTAR = "voltar_ao_ultimo_ponto_com_enlace"


@dataclass(frozen=True)
class PoliticaDaMissao:
    continuar_sem_enlace: bool = False
    atravessar_zonas_mapeadas: bool = True
    travessia_max_s: float = 120.0
    tolerancia_s: float | None = None
    voltar_apos_s: float | None = None
    fechar_aplicacao: bool = False
    origem: str = "padrão"

    @classmethod
    def de_payload(cls, payload: dict | None) -> PoliticaDaMissao:
        payload = payload or {}
        corpo = payload.get("command") if isinstance(payload.get("command"), dict) else {}
        explicita = payload.get("politica_enlace") or corpo.get("politica_enlace")
        if isinstance(explicita, dict):
            return cls(
                continuar_sem_enlace=bool(explicita.get("continuar_sem_enlace", False)),
                atravessar_zonas_mapeadas=bool(explicita.get("atravessar_zonas_mapeadas",
                                                             True)),
                travessia_max_s=float(explicita.get("travessia_max_s", 120.0)),
                tolerancia_s=_num(explicita.get("tolerancia_s")),
                voltar_apos_s=_num(explicita.get("voltar_apos_s")),
                fechar_aplicacao=bool(explicita.get("fechar_aplicacao", False)),
                origem="politica_enlace")
        offline = payload.get("offline_policy") or corpo.get("offline_policy")
        if isinstance(offline, dict):
            graca = _num(offline.get("grace_seconds")) or 120.0
            volta = graca if offline.get("after_grace") == "return_to_base" else None
            if offline.get("on_link_loss") == "pause" or offline.get("stop_application"):
                return cls(atravessar_zonas_mapeadas=False, voltar_apos_s=volta,
                           fechar_aplicacao=bool(offline.get("stop_application", True)),
                           origem="offline_policy")
            return cls(travessia_max_s=min(120.0, graca), voltar_apos_s=volta,
                       origem="offline_policy")
        return cls()


def _num(v) -> float | None:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def missao_permite_sem_enlace(payload: dict | None) -> bool:
    """``politica_enlace.continuar_sem_enlace`` no corpo assinado (ou no topo, missão local)."""
    return PoliticaDaMissao.de_payload(payload).continuar_sem_enlace


def escreve_atomico(destino: Path, dados: dict) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_name(f".{destino.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, destino)


class PoliticaDeEnlace:
    def __init__(self, no, par: str, caminho: Path | str = "/run/valtexa/enlace.json", *,
                 controlador: Any = None, missao=None, tolerancia_s: float = 2.0,
                 travessia=None, posicao=None) -> None:
        """``missao()`` devolve o corpo da missão em curso (ou ``None``); ``travessia``: a
        :class:`~.travessia.MissaoCienteDoEnlace` (perda esperada); ``posicao()``: a última
        posição (para o "último ponto com enlace")."""
        self.no = no
        self.par = par
        self.caminho = Path(caminho)
        self.controlador = controlador
        self.missao = missao or (lambda: None)
        self.tolerancia_s = tolerancia_s
        self.pausado_por_mim = False
        self.ultima_acao = ACAO_NENHUMA
        self.travessia = travessia
        self.posicao = posicao or (lambda: None)
        self.ultimo_ponto_com_enlace: dict | None = None

    def aplicar(self) -> dict:
        saude = self.no.saude()["pares"][self.par]
        parede = self.no.relogio.parede()
        desde = saude["sem_enlace_desde"]  # na hora estável do nó (imune a passo do relógio)
        sem_s = 0.0 if desde is None else max(0.0, self.no.parede() - desde)
        vivo = saude["vivo"]
        payload = self.missao()
        pol = PoliticaDaMissao.de_payload(payload)
        permite = pol.continuar_sem_enlace
        tolerancia = pol.tolerancia_s if pol.tolerancia_s is not None else self.tolerancia_s
        esperada, resta_s = (self.travessia.perda_esperada() if self.travessia is not None
                             else (False, None))
        planejada = (esperada and pol.atravessar_zonas_mapeadas
                     and sem_s <= pol.travessia_max_s)
        acao = ACAO_NENHUMA
        retorno = None
        c = self.controlador
        modo = c.mode() if c is not None and hasattr(c, "mode") else None
        if vivo:
            pos = self.posicao()
            if pos is not None:
                self.ultimo_ponto_com_enlace = {"lat": pos["lat"], "lon": pos["lon"]}
        if not vivo and sem_s >= tolerancia:
            if c is not None and hasattr(c, "teleop"):
                c.teleop(0.0, 0.0, None)
            acao = ACAO_TELEOP_ZERADO
            if modo == "manual" or (payload is not None and not permite and not planejada
                                    and modo in (None, "autonomo")):
                if c is not None and hasattr(c, "set_mode") and modo != "pausado":
                    c.set_mode("pausado")
                    self.pausado_por_mim = True
                acao = ACAO_PAUSADO
            elif payload is not None and permite:
                acao = ACAO_CONTINUA
            elif payload is not None and planejada:
                acao = ACAO_TRAVESSIA
            if self.pausado_por_mim:
                acao = ACAO_PAUSADO
                if pol.voltar_apos_s is not None and sem_s >= pol.voltar_apos_s and \
                        self.ultimo_ponto_com_enlace is not None:
                    acao = ACAO_VOLTAR  # a navegação decide como (o Runtime não move)
                    retorno = self.ultimo_ponto_com_enlace
        elif vivo and self.pausado_por_mim:
            if c is not None and hasattr(c, "set_mode") and modo == "pausado":
                c.set_mode("autonomo")
                acao = ACAO_RETOMADO
            self.pausado_por_mim = False
        self.ultima_acao = acao
        estado = {
            "at": datetime.fromtimestamp(parede, UTC).isoformat(),
            "algum_vivo": vivo,
            "sem_enlace_desde": None if desde is None else datetime.fromtimestamp(
                desde, UTC).isoformat(),
            "sem_enlace_s": round(sem_s, 1),
            "melhor": saude["melhor"],
            "enlaces": saude["enlaces"],
            "nuvem": saude["info"].get("nuvem"),
            "acao": acao,
            "missao_permite_sem_enlace": permite,
            "politica": {"origem": pol.origem, "atravessar_zonas_mapeadas":
                         pol.atravessar_zonas_mapeadas, "travessia_max_s": pol.travessia_max_s,
                         "fechar_aplicacao": pol.fechar_aplicacao},
            "fechar_aplicacao": bool(pol.fechar_aplicacao and not vivo and sem_s >= tolerancia),
            "travessia": None if self.travessia is None else {
                "estado": self.travessia.estado, "esperada": esperada,
                "resta_s": None if resta_s is None else round(resta_s, 1)},
            "retorno": retorno,
        }
        escreve_atomico(self.caminho, estado)
        return estado
