"""Rede do Rover "via base": o nó do enlace, os rádios e a política de perda de enlace.

Liga-se no ``hardware.json`` com ``"rede": {"modo": "base", ...}`` (exemplo em
``deploy/hardware.rover-v3.json.exemplo``). Sem isso, a rede continua HTTPS direto
(:class:`~valtexa_rover.hal.local.HttpNetwork`), como antes.

* cada enlace (``enlaces``) é uma interface IP de um rádio-ponte: ``bind`` local,
  ``destino`` = o IP da base naquele rádio, ``banda_bps`` = orçamento (o nominal do rádio;
  **a medir em campo**);
* a chave do enlace (``chave``) nasce no Rover na primeira vez (0600); a pública vai para a
  base assinada pela identidade (``valtexa-rover enlace-chave``);
* ``base_pub`` é a pública X25519 da base (não é segredo);
* a fila persistente do enlace fica em ``fila`` (SQLite WAL, ``synchronous=FULL``).

O que chega da base pelo canal de controle: ``{"t": "parar"}`` (difusão do console local
da base) para o robô pela navegação (``parar.json`` + modo ``parado`` + teleop zero). Parar
é sempre aceito de um par autenticado — reduz risco; **iniciar** nunca vem por aqui: missão
só com a assinatura da plataforma.

**Malha com repetidoras** (Parte 13): ``"malha": {"ativa": true, "vizinhos": {"rep-x":
{"pub": "<valtexa-base chave da repetidora>", "destinos": {"alta": "ip:porta"}}}}``. O Rover
inicia com a base e com cada repetidora que ouve, e fala com a base pela
:class:`~valtexa_rover.enlace.malha.Malha` (rota pela qualidade medida, troca sozinha,
confirmação fim a fim). O resto do Runtime não muda: o "par" continua sendo ``"base"``.

**Não testado em hardware** (rádio real): só com o emulador e UDP no ``localhost``.
"""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from pathlib import Path
from typing import Any

from valtexa_rover.enlace.cripto import ChaveDeEnlace
from valtexa_rover.enlace.fila import FilaCheiaError, FilaPersistente
from valtexa_rover.enlace.levantamento import (
    MapaDeCobertura,
    RegistradorDeCobertura,
    posicao_de,
)
from valtexa_rover.enlace.malha import Malha
from valtexa_rover.enlace.no import P1, ConfigDoEnlace, NoDeEnlace
from valtexa_rover.enlace.politica import PoliticaDeEnlace
from valtexa_rover.enlace.rpc import CANAL_CONTROLE, ClienteRpc, Executor, RedeViaBase
from valtexa_rover.enlace.transporte import Relogio, TransporteUdp, endereco
from valtexa_rover.enlace.travessia import MissaoCienteDoEnlace, rota_da_missao

log = logging.getLogger("valtexa.enlace")

PAR_BASE = "base"


class ServicoDeEnlace:
    """O enlace do lado do Rover: nó + thread + RPC + política + canal de controle."""

    def __init__(self, cfg: dict, dados: Path, *, relogio=None, transportes=None) -> None:
        self.cfg = cfg
        self.relogio = relogio or Relogio()
        chave = ChaveDeEnlace.carregar_ou_criar(cfg.get("chave") or Path(dados) / "enlace.key")
        enlaces = []
        for i, e in enumerate(cfg["enlaces"]):
            t = (transportes[i] if transportes else
                 TransporteUdp(e["nome"], endereco(e["bind"]), mtu=int(e.get("mtu", 1200)),
                               interface=e.get("interface")))
            destino = e.get("destino")
            enlaces.append(ConfigDoEnlace(e["nome"], t, float(e["banda_bps"]),
                                          endereco(destino) if isinstance(destino, str)
                                          and ":" in destino else destino))
        fila = FilaPersistente(cfg.get("fila") or Path(dados) / "enlace.db", self.relogio.parede,
                               sincrono=cfg.get("sincrono", "FULL"))
        malha = cfg.get("malha") or {}
        base_pub = bytes.fromhex(cfg["base_pub"])
        nome = cfg.get("nome", "rover")
        if malha.get("ativa"):
            vizinhos = malha.get("vizinhos") or {}
            pares = {PAR_BASE: base_pub, **{n: bytes.fromhex(v["pub"])
                                            for n, v in vizinhos.items()}}
            destinos = {PAR_BASE: {e.nome: e.destino for e in enlaces if e.destino is not None}}
            for n, v in vizinhos.items():
                destinos[n] = {enl: endereco(d) if isinstance(d, str) else d
                               for enl, d in (v.get("destinos") or {}).items()}
            self.no = NoDeEnlace(nome, chave, fila, self.relogio, enlaces, pares,
                                 iniciador=False, info_batimento=self._info,
                                 destinos=destinos)
            self.canal = Malha(nome, self.no, cfg.get("malha_fila") or Path(dados) / "malha.db",
                               atributos=lambda: {"papel": "rover"},
                               publicas={PAR_BASE: base_pub},
                               sincrono=cfg.get("sincrono", "FULL"))
        else:
            self.no = NoDeEnlace(nome, chave, fila, self.relogio, enlaces, {PAR_BASE: base_pub},
                                 iniciador=True, info_batimento=self._info)
            self.canal = self.no
        """Por onde o Runtime fala com a base: o nó (um salto) ou a malha (vários)."""
        prazos = {int(k): float(v) for k, v in (cfg.get("prazos_s") or {}).items()}
        self.cliente = ClienteRpc(self.canal, PAR_BASE, prazos)
        self.rede = RedeViaBase(self.cliente, chave_plataforma=cfg.get("plataforma_pub"),
                                exige_recibo=bool(cfg.get("exige_recibo", True)))
        self.rede.servico = self  # type: ignore[attr-defined]
        self.controlador: Any = None
        self.missao_atual: dict | None = None
        """O laço principal atualiza (a política não lê o banco do Runtime de outra thread)."""
        self.politica: PoliticaDeEnlace | None = None
        self.posicao: dict | None = None
        """Última posição do GNSS (o laço principal atualiza; vai no batimento para a base)."""
        self._respostas: dict[str, dict] = {}
        self._resposta_chegou = threading.Condition()
        self.sync_pendente = 0
        """Itens na fila de sincronização do Runtime (o laço principal atualiza)."""
        self.travessia: MissaoCienteDoEnlace | None = None
        self._dados = Path(dados)
        cobertura = cfg.get("cobertura")
        self._mapa_do_arquivo: MapaDeCobertura | None = (
            MapaDeCobertura.carregar(cobertura) if cobertura and Path(cobertura).exists()
            else None)
        """O ``cobertura.json`` local (instalação antiga, ou sem plataforma): vale quando a
        missão não traz o mapa."""
        self._marca_do_mapa: str | None = None
        if self._mapa_do_arquivo is not None:
            self.travessia = self._nova_travessia(self._mapa_do_arquivo)
        lv = cfg.get("levantamento") or {}
        self.registrador: RegistradorDeCobertura | None = None
        if lv.get("ativo"):
            self.registrador = RegistradorDeCobertura(
                lv.get("arquivo") or Path(dados) / "levantamento.jsonl", lado="rover",
                no=nome, saude=self._saude_vizinhos, posicao=lambda: self.posicao,
                intervalo_s=float(lv.get("intervalo_s", 1.0)),
                exige_rtk=bool(lv.get("exige_rtk", True)), parede=self.relogio.parede)
        self.paradas_da_base = 0
        self._ultima_politica = -1e9
        self._trava = threading.Lock()
        self.executor = Executor(self.canal, ao_passo=self._ao_passo)

    # -- ciclo de vida -------------------------------------------------------------

    def iniciar(self) -> ServicoDeEnlace:
        self.executor.iniciar()
        return self

    def parar(self) -> None:
        self.executor.parar()
        self.canal.fechar()

    def liga(self, controlador: Any) -> None:
        """Depois que o Hardware existe: a política e o PARAR da base agem na navegação."""
        self.controlador = controlador
        self.politica = PoliticaDeEnlace(
            self.canal, PAR_BASE, self.cfg.get("estado", "/run/valtexa/enlace.json"),
            controlador=controlador, missao=lambda: self.missao_atual,
            tolerancia_s=float(self.cfg.get("tolerancia_s", 2.0)), travessia=self.travessia,
            posicao=lambda: self.posicao)

    def _nova_travessia(self, mapa: MapaDeCobertura) -> MissaoCienteDoEnlace:
        tv = self.cfg.get("travessia") or {}
        return MissaoCienteDoEnlace(
            mapa, registro=tv.get("registro") or self._dados / "travessias.jsonl",
            parede=self.canal.parede, antecedencia_s=float(tv.get("antecedencia_s", 20)),
            v_padrao_ms=float(tv.get("v_padrao_ms", 0.8)), avisar=self._avisa_travessia)

    def cobertura_da_missao(self, mapa: dict | None) -> bool:
        """O mapa de cobertura que veio **assinado na missão** (a plataforma guarda o último
        levantamento da base por talhão). Troca o mapa da travessia quando muda; sem mapa
        na missão, volta ao ``cobertura.json`` local (se houver). Mapa malformado não
        derruba nada: fica o anterior e o log diz. ``True`` se trocou."""
        marca = None if not mapa else json.dumps(
            [mapa.get("gerado_em"), mapa.get("celula_m"), mapa.get("origem"),
             len(mapa.get("celulas") or {})], sort_keys=True)
        if marca == self._marca_do_mapa:
            return False
        if mapa:
            try:
                novo: MapaDeCobertura | None = MapaDeCobertura(mapa)
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                log.warning("mapa de cobertura da missão inválido: %s", exc)
                self._marca_do_mapa = marca  # não tenta de novo a cada volta
                return False
        else:
            novo = self._mapa_do_arquivo
        self._marca_do_mapa = marca
        if novo is None:
            if self.travessia is not None:
                self.travessia.mapa = None  # sem mapa: a política volta ao padrão
            return True
        if self.travessia is None:
            self.travessia = self._nova_travessia(novo)
            if self.politica is not None:
                self.politica.travessia = self.travessia
        else:
            self.travessia.mapa = novo
        log.info("mapa de cobertura %s: %d células, %d zona(s) morta(s)",
                 "da missão" if mapa else "local", len(novo.celulas), len(novo.zonas))
        return True

    def pergunta_base(self, pedido: dict, timeout_s: float = 3.0) -> dict | None:
        """Pedido de controle à base (P1) e a resposta dela, ou ``None`` no prazo.

        Usado pelos ensaios de enlace: ``eco`` (latência de P1 no rádio, sem a nuvem),
        ``relogio`` (chrony da base) e — só com ``ensaios_remotos`` ligado na base —
        ``silenciar`` (um rádio "morre" por N s), ``sem_internet`` (a Starlink "cai" por N s)
        e ``reiniciar`` (o serviço da base cai e volta, como numa queda de energia)."""
        rid = secrets.token_hex(8)
        try:
            self.canal.enviar(PAR_BASE, json.dumps({**pedido, "id": rid}).encode(), P1,
                              CANAL_CONTROLE, ttl_s=timeout_s)
        except FilaCheiaError:
            return None
        fim = time.monotonic() + timeout_s
        with self._resposta_chegou:
            while rid not in self._respostas:
                resta = fim - time.monotonic()
                if resta <= 0:
                    return None
                self._resposta_chegou.wait(min(resta, 0.05))
            return self._respostas.pop(rid)

    def atualiza_posicao(self, fix) -> None:
        self.posicao = posicao_de(fix)

    def _avisa_travessia(self, aviso: dict) -> None:
        """P1 à base: "vou entrar numa zona morta mapeada, volto em ~N s"."""
        try:
            self.canal.enviar(PAR_BASE, json.dumps(aviso).encode(), P1, CANAL_CONTROLE,
                              ttl_s=30.0)
        except FilaCheiaError:
            log.warning("aviso de travessia não coube na fila do enlace")

    def _atualiza_travessia(self) -> None:
        pos = self.posicao
        filas = self.no.fila.tamanho_saida()
        vazio = self.sync_pendente == 0 and not any(n for p, n in filas.items() if p >= 2)
        self.travessia.atualiza(
            pos, rota=rota_da_missao(self.missao_atual),
            v_ms=None if pos is None else pos.get("v_ms"),
            rumo_deg=None if pos is None else pos.get("rumo_deg"),
            vivo=self.canal.vivo(PAR_BASE), filas={f"P{k}": v for k, v in filas.items()},
            vazio=vazio)

    def _saude_vizinhos(self) -> dict:
        """Cada vizinho de rádio (a base e, na malha, as repetidoras) com os enlaces dele."""
        return self.no.saude()

    # -- na thread do enlace ---------------------------------------------------------

    def _info(self) -> dict:
        c = self.controlador
        modo = None
        if c is not None and hasattr(c, "mode"):
            try:
                modo = c.mode()
            except (OSError, ValueError):
                modo = None
        info = {"modo": modo, "missao": (self.missao_atual or {}).get("mission_id"),
                "acao_enlace": self.politica.ultima_acao if self.politica else None}
        pos = self.posicao
        if pos is not None:  # a base registra a cobertura do lado dela com esta posição
            info["pos"] = [round(pos["lat"], 7), round(pos["lon"], 7), pos.get("fix")]
        return info

    def _ao_passo(self) -> None:
        for c in self.canal.chegadas(CANAL_CONTROLE, 16):
            self.canal.consumida(c.seq)
            try:
                msg = json.loads(c.corpo)
            except ValueError:
                continue
            if msg.get("t") == "parar":
                self.parar_pela_base(str(msg.get("motivo") or "PARAR do console da base"))
            elif msg.get("t") == "resp" and msg.get("id"):
                with self._resposta_chegou:
                    self._respostas[str(msg["id"])] = {**msg, "_chegou": self.relogio.agora()}
                    while len(self._respostas) > 256:
                        self._respostas.pop(next(iter(self._respostas)))
                    self._resposta_chegou.notify_all()
        agora = self.relogio.agora()
        if self.politica is not None and agora - self._ultima_politica >= 0.5:
            self._ultima_politica = agora
            if self.travessia is not None:
                try:
                    self._atualiza_travessia()
                except OSError as exc:  # registro da travessia: nunca derruba o enlace
                    log.warning("travessia não registrada: %s", exc)
            try:
                self.politica.aplicar()
            except OSError as exc:  # /run/valtexa indisponível: registra e segue
                log.warning("enlace.json não gravado: %s", exc)
        if self.registrador is not None:
            try:
                self.registrador.talvez()
            except OSError as exc:  # disco: o levantamento nunca derruba o enlace
                log.warning("levantamento não gravado: %s", exc)

    def parar_pela_base(self, motivo: str) -> None:
        self.paradas_da_base += 1
        c = self.controlador
        log.warning("PARAR recebido da base: %s", motivo)
        if c is None:
            return
        if hasattr(c, "set_mode"):
            c.set_mode("parado")
        if hasattr(c, "stop"):  # depois do modo: o motivo que fica no parar.json é o da base
            c.stop(f"base: {motivo}")
        if hasattr(c, "teleop"):
            c.teleop(0.0, 0.0, None)


def rede_via_base(cfg: dict, dados: Path, **kw) -> RedeViaBase:
    """A HAL de rede do ``hardware.json`` com ``"modo": "base"`` (o serviço já rodando)."""
    return ServicoDeEnlace(cfg, dados, **kw).iniciar().rede
