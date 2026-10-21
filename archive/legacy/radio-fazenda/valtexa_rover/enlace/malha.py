"""Malha com repetidoras (Parte 13): vários saltos, rota pela qualidade medida, troca de rota
quando um nó morre, guarda-e-encaminha e confirmação **fim a fim**.

Cada nó (Rover, repetidora solar, base) roda um :class:`~.no.NoDeEnlace` com os
**vizinhos** como pares (criptografia por salto, ARQ, filas persistentes, P0-P4 — tudo da
P12) e, por cima, uma :class:`Malha`, que leva cada mensagem da origem ao destino final::

    Rover ──salto──▶ repetidora ──salto──▶ repetidora ──salto──▶ base
      │  cada salto: entrega confiável e persistente (a repetidora guarda e encaminha)
      └─ fim a fim: a origem guarda a cópia até o destino confirmar ("A"); se o caminho
         morre no meio, reenvia por outro; o destino deduplica pelo id (uma vez só)

**Rota pela qualidade medida** (vetor de distâncias): o custo de um enlace sai da saúde
medida pelo próprio protocolo — RTT, perda e banda (:func:`custo_do_enlace`, em segundos:
"quanto custa levar um pacote") — e o de uma rota é a soma dos saltos + ``custo_salto_s``
por salto. Cada nó anuncia no batimento (0,5 s) o custo dele até cada destino, com
**reverso envenenado** (não anuncia a quem é o próximo salto daquela rota) e teto de
:data:`MAX_SALTOS`. Anúncio velho (> ``anuncio_validade_s``) não vale: vizinho morto some
das rotas em ~3 s. Troca de rota com histerese: na hora se o próximo salto morreu ou se a
nova custa menos da metade; senão, só se custar < 80 % por 2 s seguidos.

**Redundância**: P0/P1 saem pelas **duas** melhores rotas com próximos saltos diferentes (o
destino deduplica) — a queda de uma repetidora no meio de um comando não atrasa nada.

**Repetidora com bateria fraca** (``custo_extra``) anuncia custo maior: as rotas fogem dela
antes que ela apague.

**Teleop** (*slots*) atravessa a malha "só o último vale" em cada salto, com a sequência da
origem conferida no destino (ponto velho que chega por outro caminho não é aplicado).

A :class:`Malha` tem a mesma cara do nó para quem está em cima (``enviar``,
``enviar_ultimo``, ``chegadas``, ``consumida``, ``ultimo``, ``vivo``, ``info_do_par``,
``saude``, ``pares``, ``parede``, ``passo``): o RPC (:mod:`.rpc`), a política de perda de
enlace e a base funcionam sobre ela sem mudar — o "par" passa a ser o destino final.

**Simulado / não testado em hardware**: rádios e repetidoras só no emulador; o custo dos
enlaces e os tempos padrão são hipóteses a ajustar com o levantamento de campo.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
import struct
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from valtexa_rover.enlace.fila import Chegada, FilaCheiaError
from valtexa_rover.enlace.no import P1, P4, NoDeEnlace
from valtexa_rover.enlace.saude import SaudeDoEnlace

CANAL_MALHA = 9
MAX_SALTOS = 8
PENALIDADE_SUSPEITO_S = 2.0
SEM_ROTA = -1

ESQUEMA = """
CREATE TABLE IF NOT EXISTS malha_saida (
    id BLOB PRIMARY KEY, dest TEXT NOT NULL, prio INTEGER NOT NULL, canal INTEGER NOT NULL,
    corpo BLOB NOT NULL, expira REAL, criado REAL NOT NULL, tamanho INTEGER NOT NULL,
    ordem INTEGER NOT NULL, salto TEXT, salto_id BLOB, salto_ok REAL, tentativas INTEGER
    NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS malha_saida_salto ON malha_saida (salto_id);
"""


def custo_do_enlace(s: SaudeDoEnlace) -> float | None:
    """Segundos esperados para levar um pacote de 1200 B por este enlace (``None`` = morto).

    ``(serialização + meio RTT + fixo) / probabilidade de passar``: perda alta multiplica
    (é o ETX), RTT alto e banda baixa somam. Hipótese simples e explicável — os pesos se
    ajustam com o levantamento de campo.
    """
    if not s.vivo:
        return None
    passa = max(0.05, 1.0 - min(0.95, s.perda))
    return (0.01 + (s.srtt if s.srtt is not None else 0.2) / 2 + 9600 / s.banda_bps) / passa


def _hkdf_ack(dh: bytes, a: str, b: str) -> bytes:
    """Chave das confirmações fim a fim entre ``a`` e ``b`` (a mesma dos dois lados)."""
    x, y = sorted((a, b))
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=b"vx-malha-ack-v1",
                info=f"{x}\n{y}".encode()).derive(dh)


def _assina_ack(chave: bytes, ids: bytes) -> str:
    return hmac.new(chave, ids, hashlib.sha256).hexdigest()[:32]


def _codifica(cab: dict, corpo: bytes = b"") -> bytes:
    j = json.dumps(cab, separators=(",", ":"), ensure_ascii=False).encode()
    return struct.pack("!H", len(j)) + j + corpo


def _decodifica(dados: bytes) -> tuple[dict, bytes]:
    (n,) = struct.unpack_from("!H", dados)
    return json.loads(dados[2:2 + n]), dados[2 + n:]


@dataclass
class ParametrosDaMalha:
    custo_salto_s: float = 0.03
    """Cada salto a mais custa isto (prefere menos saltos com qualidade parecida)."""
    troca_fator: float = 0.8
    troca_espera_s: float = 2.0
    troca_imediata: float = 0.5
    anuncio_validade_s: float = 3.0
    reavalia_s: float = 0.25
    redundancia_p01: bool = True
    ack_lote_s: float = 0.1
    e2e_base_s: dict[int, float] = field(
        default_factory=lambda: {0: 2.0, 1: 2.0, 2: 15.0, 3: 15.0, 4: 90.0})
    """Espera da confirmação fim a fim (depois de o 1º salto guardar), antes de reenviar."""
    limite_bytes: int = 512 * 1024 * 1024
    reserva_p01_bytes: int = 32 * 1024 * 1024
    lembra_encaminhados: int = 5000


@dataclass
class Rota:
    dest: str
    prox: str
    custo: float
    saltos: int
    atributos: dict = field(default_factory=dict)
    desde: float = 0.0


@dataclass
class _Candidato:
    prox: str
    custo: float
    saltos: int
    atributos: dict


@dataclass
class _Despacho:
    """Uma cópia entregue à fila de um salto (para trocar de rota se o vizinho morrer)."""

    e2e: bytes
    dest: str
    vizinho: str
    prio: int
    canal_cab: dict
    corpo: bytes | None
    expira: float | None
    origem_aqui: bool


class Malha:
    def __init__(self, nome: str, no: NoDeEnlace, caminho: Path | str, *,
                 parametros: ParametrosDaMalha | None = None,
                 atributos: Callable[[], dict] | None = None,
                 custo_extra: Callable[[], float] | None = None,
                 publicas: dict[str, bytes] | None = None,
                 sincrono: str = "FULL") -> None:
        """``atributos()``: o que este nó anuncia sobre si (a base: ``{"nuvem": bool}``);
        ``custo_extra()``: segundos somados a toda rota que passa por aqui (bateria fraca);
        ``publicas``: chaves X25519 de enlace das **pontas** (o Rover conhece a da base e
        vice-versa) — a confirmação fim a fim vai autenticada por elas (HMAC com o DH
        estático), e uma repetidora não consegue forjar "chegou" para a origem apagar a
        cópia."""
        self.nome = nome
        self.no = no
        self.relogio = no.relogio
        self.novidade = no.novidade
        self.estat = no.estat
        self.p = parametros or ParametrosDaMalha()
        self.atributos = atributos or (lambda: {})
        self.custo_extra = custo_extra or (lambda: 0.0)
        self._trava: threading.RLock = no._trava
        self._info_base = no.info_batimento
        no.info_por_par = self._anuncio_para
        no.ao_entregue = self._salto_entregue
        self._db = sqlite3.connect(caminho, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        if sincrono not in ("FULL", "NORMAL"):
            raise ValueError("sincrono: FULL ou NORMAL")
        self._db.execute(f"PRAGMA synchronous={sincrono}")
        self._db.executescript(ESQUEMA)
        self._ordem = self._db.execute("SELECT COALESCE(MAX(ordem), 0) FROM malha_saida"
                                       ).fetchone()[0]
        self.rotas: dict[str, Rota] = {}
        self.segundas: dict[str, Rota] = {}
        self._candidato_troca: dict[str, tuple[str, float]] = {}
        self.sem_rota_desde: dict[str, float] = {}
        self.conhecidos: set[str] = set(no.pares())
        self._desp: dict[bytes, _Despacho] = {}
        self._acks: dict[str, list[bytes]] = {}
        self._acks_em = 0.0
        self._reavalia_em = -1e9
        self._slot_seq_rx: dict[tuple[str, str], int] = {}
        self._tentativa: dict[bytes, int] = {}
        self._passou: OrderedDict[tuple[str, int], None] = OrderedDict()
        """(id, tentativa) que já passaram por aqui: a mesma cópia voltando = laço, cai."""
        self._slots_rx: dict[tuple[str, str], Chegada] = {}
        self.trocas: list[dict] = []
        self.trocas_por_destino: dict[str, int] = {}
        """Cada troca de rota: ``{"dest", "de", "para", "t", "motivo"}`` (relatório)."""
        for k in ("malha_entregues", "malha_repetidas", "malha_encaminhadas", "malha_reenvios",
                  "malha_sem_rota", "malha_saltos_demais", "malha_acks", "malha_vencidas",
                  "malha_trocas"):
            self.estat.setdefault(k, 0)
        self._e2e_chaves: dict[str, bytes] = {}
        for outro, pub in (publicas or {}).items():
            self._e2e_chaves[outro] = _hkdf_ack(no.chave.dh(pub), nome, outro)

    # ------------------------------------------------------------------ cara do nó

    @property
    def enlaces(self):
        return self.no.enlaces

    def parede(self) -> float:
        return self.no.parede()

    def pares(self) -> list[str]:
        with self._trava:
            return sorted(self.conhecidos - {self.nome})

    def vivo(self, dest: str) -> bool:
        with self._trava:
            return dest in self.rotas

    def info_do_par(self, dest: str) -> dict:
        with self._trava:
            r = self.rotas.get(dest)
            return dict(r.atributos) if r is not None else {}

    def chegadas(self, canal: int | None = None, limite: int = 100) -> list[Chegada]:
        if canal == CANAL_MALHA:
            return []
        return self.no.chegadas(canal, limite, excluir_canal=CANAL_MALHA)

    def consumida(self, seq: int) -> None:
        self.no.consumida(seq)

    def slots_recebidos(self, origem: str) -> list[str]:
        """Os slots fim a fim chegados de ``origem`` (teleop, vídeo de cada câmera)."""
        with self._trava:
            return [slot for (o, slot) in self._slots_rx if o == origem]

    def ultimo(self, origem: str, slot: str) -> Chegada | None:
        with self._trava:
            c = self._slots_rx.pop((origem, slot), None)
        if c is None or (c.expira is not None and c.expira <= self.parede()):
            return None
        return c

    def fechar(self) -> None:
        self.no.fechar()
        with self._trava:
            self._db.close()

    def saude(self) -> dict:
        with self._trava:
            vizinhos = self.no.saude()
            pares = {}
            for dest in sorted(self.conhecidos - {self.nome}):
                r = self.rotas.get(dest)
                prox = r.prox if r else None
                pares[dest] = {
                    "vivo": r is not None,
                    "melhor": prox,
                    "rota": None if r is None else {"prox": prox, "custo_ms": round(
                        r.custo * 1000, 1), "saltos": r.saltos},
                    "segunda": None if dest not in self.segundas else self.segundas[dest].prox,
                    "enlaces": (vizinhos["pares"].get(prox, {}).get("enlaces", {})
                                if prox else {}),
                    "sem_enlace_desde": self.sem_rota_desde.get(dest),
                    "trocas": sum(1 for t in self.trocas if t["dest"] == dest),
                    "info": dict(r.atributos) if r else {},
                    "em_voo": self._pendentes_para(dest),
                }
            return {"no": self.nome, "pares": pares, "vizinhos": vizinhos["pares"],
                    "fila": vizinhos["fila"], "descartes": vizinhos["descartes"],
                    "estat": dict(self.estat)}

    def _pendentes_para(self, dest: str) -> int:
        return self._db.execute("SELECT COUNT(*) FROM malha_saida WHERE dest = ?",
                                (dest,)).fetchone()[0]

    # ------------------------------------------------------------------ envio

    def enviar(self, dest: str, corpo: bytes, prio: int, canal: int = 0,
               ttl_s: float | None = None) -> bytes:
        """Entrega confiável fim a fim. A cópia fica aqui até o destino confirmar."""
        if not 0 <= prio <= 4:
            raise ValueError("prioridade de 0 a 4")
        e2e = secrets.token_bytes(16)
        expira = None if ttl_s is None else self.parede() + ttl_s
        n = len(corpo)
        with self._trava:
            if prio <= P1:
                if self._bytes("prio <= 1") + n > self.p.reserva_p01_bytes:
                    raise FilaCheiaError("reserva de P0/P1 da malha cheia")
            elif self._bytes("prio >= 2") + n > self.p.limite_bytes:
                raise FilaCheiaError("fila fim a fim da malha cheia")
            self._ordem += 1
            self._db.execute(
                "INSERT INTO malha_saida (id, dest, prio, canal, corpo, expira, criado, tamanho,"
                " ordem) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (e2e, dest, prio, canal, corpo, expira, self.parede(), n, self._ordem))
            self._despacha(e2e, dest, prio, canal, corpo, expira)
        return e2e

    def _bytes(self, onde: str) -> int:
        return self._db.execute(
            f"SELECT COALESCE(SUM(tamanho), 0) FROM malha_saida WHERE {onde}").fetchone()[0]

    def enviar_ultimo(self, dest: str, slot: str, corpo: bytes, ttl_s: float, *,
                      prio: int = P1, canal: int = 0) -> None:
        with self._trava:
            sq = self.no.proxima_seq_slot(f"malha:{dest}:{slot}")
            cab = {"t": "s", "o": self.nome, "d": dest, "s": 1, "c": canal, "sl": slot,
                   "sq": sq}
            for viz in self._proximos(dest, prio):
                self.no.enviar_ultimo(viz, f"{self.nome}>{dest}:{slot}", _codifica(cab, corpo),
                                      ttl_s, prio=prio, canal=CANAL_MALHA)

    def _proximos(self, dest: str, prio: int, evitar: str | None = None,
                  redundante: bool = True) -> list[str]:
        """Próximo(s) salto(s). P0/P1 vão por dois caminhos **só na origem** (no meio do
        caminho, duplicar a cada salto multiplicaria as cópias)."""
        r = self.rotas.get(dest)
        if r is None or r.prox == evitar:
            r = self.segundas.get(dest) if dest in self.segundas and \
                self.segundas[dest].prox != evitar else None
            return [] if r is None else [r.prox]
        saida = [r.prox]
        seg = self.segundas.get(dest)
        if redundante and prio <= P1 and self.p.redundancia_p01 and seg is not None and \
                seg.prox not in (r.prox, evitar):
            saida.append(seg.prox)
        return saida

    def _despacha(self, e2e: bytes, dest: str, prio: int, canal: int, corpo: bytes,
                  expira: float | None) -> bool:
        """Põe a mensagem (origem aqui) na fila do(s) próximo(s) salto(s)."""
        vizinhos = self._proximos(dest, prio)
        if not vizinhos:
            self.estat["malha_sem_rota"] += 1
            return False
        resta = None if expira is None else expira - self.parede()
        if resta is not None and resta <= 0:
            return False
        self._tentativa[e2e] = self._tentativa.get(e2e, 0) + 1
        cab = {"t": "d", "o": self.nome, "d": dest, "id": e2e.hex(), "s": 1, "c": canal,
               "n": self._tentativa[e2e]}
        quadro = _codifica(cab, corpo)
        feito = False
        for viz in vizinhos:
            try:
                sid = self.no.enviar(viz, quadro, prio, CANAL_MALHA, ttl_s=resta)
            except FilaCheiaError:
                continue
            self._desp[sid] = _Despacho(e2e, dest, viz, prio, cab, None, expira, True)
            self._db.execute(
                "UPDATE malha_saida SET salto = ?, salto_id = ?, salto_ok = NULL, "
                "tentativas = tentativas + 1 WHERE id = ?", (viz, sid, e2e))
            feito = True
        return feito

    # ------------------------------------------------------------------ laço

    def passo(self) -> None:
        self.no.passo()
        self.processa()

    def processa(self) -> None:
        """Depois do ``passo`` do nó: rotas, mensagens da malha, confirmações, reenvios."""
        with self._trava:
            agora = self.relogio.agora()
            if agora - self._reavalia_em >= self.p.reavalia_s:
                self._reavalia_em = agora
                self._atualiza_rotas(agora)
                self._reenvios(agora)
            self._recebe_malha()
            if self._acks and agora - self._acks_em >= self.p.ack_lote_s:
                self._acks_em = agora
                self._manda_acks()

    # -- rotas ----------------------------------------------------------------------

    def _anuncios(self, agora: float) -> dict[str, dict]:
        saida = {}
        for viz in self.no.pares():
            p = self.no._pares[viz]
            if p.info_em is None or agora - p.info_em > self.p.anuncio_validade_s:
                continue
            m = (p.info or {}).get("malha")
            if isinstance(m, dict):
                saida[viz] = m
        return saida

    def _atualiza_rotas(self, agora: float) -> None:
        anuncios = self._anuncios(agora)
        candidatos: dict[str, list[_Candidato]] = {}
        for viz in self.no.pares():
            # calado há 1 s (suspeito): custa caro — a rota já vai por outro vizinho que
            # está falando, sem esperar os 3 s da morte; se não houver outro, fica
            custos = [c + (PENALIDADE_SUSPEITO_S if s.suspeito(agora) else 0.0)
                      for s in self.no._pares[viz].saude.values()
                      if (c := custo_do_enlace(s)) is not None]
            if not custos:
                continue
            cl = min(custos)
            anuncio = anuncios.get(viz, {})
            proprio = (anuncio.get("r") or {}).get(viz)
            atrib_viz = proprio[2] if isinstance(proprio, list) and len(proprio) > 2 else {}
            candidatos.setdefault(viz, []).append(_Candidato(viz, cl, 1, atrib_viz or {}))
            for dest, v in (anuncio.get("r") or {}).items():
                self.conhecidos.add(dest)
                if dest in (self.nome, viz) or not isinstance(v, list) or len(v) < 2:
                    continue
                custo_ms, saltos = v[0], v[1]
                if custo_ms is None or custo_ms < 0 or saltos + 1 > MAX_SALTOS:
                    continue
                candidatos.setdefault(dest, []).append(_Candidato(
                    viz, cl + custo_ms / 1000 + self.p.custo_salto_s, saltos + 1,
                    v[2] if len(v) > 2 and isinstance(v[2], dict) else {}))
        parede = self.parede()
        for dest in self.conhecidos - set(candidatos) - set(self.rotas) - {self.nome}:
            self.sem_rota_desde.setdefault(dest, parede)  # nunca teve rota: conta desde já
        for dest in set(self.rotas) | set(candidatos):
            lista = sorted(candidatos.get(dest, []), key=lambda c: c.custo)
            atual = self.rotas.get(dest)
            if not lista:
                if atual is not None:
                    self._troca(dest, atual.prox, None, agora, "sem rota")
                    del self.rotas[dest]
                    self.sem_rota_desde.setdefault(dest, parede)
                self.segundas.pop(dest, None)
                continue
            self.sem_rota_desde.pop(dest, None)
            melhor = lista[0]
            pelo_atual = next((c for c in lista if atual and c.prox == atual.prox), None)
            escolhido = melhor
            if pelo_atual is not None and melhor.prox != pelo_atual.prox:
                if melhor.custo < self.p.troca_imediata * pelo_atual.custo:
                    self._candidato_troca.pop(dest, None)
                elif melhor.custo < self.p.troca_fator * pelo_atual.custo:
                    cand = self._candidato_troca.get(dest)
                    if cand is None or cand[0] != melhor.prox:
                        self._candidato_troca[dest] = (melhor.prox, agora)
                        escolhido = pelo_atual
                    elif agora - cand[1] < self.p.troca_espera_s:
                        escolhido = pelo_atual
                    else:
                        self._candidato_troca.pop(dest, None)
                else:
                    self._candidato_troca.pop(dest, None)
                    escolhido = pelo_atual
            if atual is None or atual.prox != escolhido.prox:
                motivo = ("nova" if atual is None else
                          "próximo salto morreu" if pelo_atual is None else "melhor custo")
                self._troca(dest, None if atual is None else atual.prox, escolhido.prox, agora,
                            motivo)
                self.rotas[dest] = Rota(dest, escolhido.prox, escolhido.custo, escolhido.saltos,
                                        escolhido.atributos, agora)
            else:
                atual.custo, atual.saltos = escolhido.custo, escolhido.saltos
                atual.atributos = escolhido.atributos
            outra = next((c for c in lista if c.prox != escolhido.prox), None)
            if outra is None:
                self.segundas.pop(dest, None)
            else:
                self.segundas[dest] = Rota(dest, outra.prox, outra.custo, outra.saltos,
                                           outra.atributos, agora)

    def _troca(self, dest: str, de: str | None, para: str | None, agora: float,
               motivo: str) -> None:
        self.trocas.append({"dest": dest, "de": de, "para": para, "t": agora, "motivo": motivo})
        if de is not None:
            self.estat["malha_trocas"] += 1
            self.trocas_por_destino[dest] = self.trocas_por_destino.get(dest, 0) + 1
        del self.trocas[:-500]

    def _anuncio_para(self, vizinho: str) -> dict:
        extra = max(0.0, float(self.custo_extra()))
        rotas: dict[str, list] = {self.nome: [round(extra * 1000), 0, self.atributos()]}
        for dest, r in self.rotas.items():
            if dest == vizinho:
                continue
            if r.prox == vizinho:  # reverso envenenado: não ensina ao vizinho a volta por ele
                rotas[dest] = [SEM_ROTA, 0]
            else:
                item = [round((r.custo + extra) * 1000), r.saltos]
                if r.atributos:
                    item.append(r.atributos)
                rotas[dest] = item
        return {**self._info_base(), "malha": {"r": rotas}}

    # -- recepção -------------------------------------------------------------------

    def _recebe_malha(self) -> None:
        for c in self.no.chegadas(CANAL_MALHA, 500):
            try:
                cab, corpo = _decodifica(c.corpo)
            except (ValueError, struct.error):
                self.no.consumida(c.seq)
                continue
            if self._trata(cab, corpo, c.prio, c.expira, c.par):
                self.no.consumida(c.seq)
        for viz in self.no.pares():
            for slot in self.no.slots_recebidos(viz):
                c = self.no.ultimo(viz, slot)
                if c is None:
                    continue
                try:
                    cab, corpo = _decodifica(c.corpo)
                except (ValueError, struct.error):
                    continue
                self._trata_slot(cab, corpo, c, viz)

    def _trata(self, cab: dict, corpo: bytes, prio: int, expira: float | None,
               de: str) -> bool:
        """``True`` = pode consumir (entregue, encaminhado ou descartado)."""
        dest, origem = str(cab.get("d")), str(cab.get("o"))
        self.conhecidos.add(origem)
        if expira is not None and expira <= self.parede():
            self.estat["malha_vencidas"] += 1
            return True
        if dest == self.nome:
            if cab.get("t") == "a":
                chave = self._e2e_chaves.get(origem)
                if chave is not None and not hmac.compare_digest(
                        str(cab.get("h", "")), _assina_ack(chave, corpo)):
                    self.estat["malha_ack_forjado"] = self.estat.get("malha_ack_forjado", 0) + 1
                    return True
                for i in range(0, len(corpo), 16):
                    self._confirmado(corpo[i:i + 16])
                return True
            e2e = bytes.fromhex(str(cab.get("id")))
            try:
                novo = self.no.fila.recebe(origem, e2e, prio, int(cab.get("c", 0)), corpo,
                                           expira)
            except sqlite3.Error:  # disco cheio: fica na entrada do salto e tenta depois
                return False
            if novo:
                self.estat["malha_entregues"] += 1
                self.novidade.notify_all()
            else:
                self.estat["malha_repetidas"] += 1
            if int(cab.get("s", 1)) >= 2:  # passou por repetidora: confirma à origem
                self._acks.setdefault(origem, []).append(e2e)
            return True
        return self._encaminha(cab, corpo, prio, expira, de)

    def _encaminha(self, cab: dict, corpo: bytes, prio: int, expira: float | None,
                   de: str, reenvio: bool = False) -> bool:
        saltos = int(cab.get("s", 1))
        if saltos >= MAX_SALTOS:
            self.estat["malha_saltos_demais"] += 1
            return True
        marca = (str(cab.get("id", "")) + str(cab.get("t")), int(cab.get("n", 0)))
        if cab.get("t") == "d" and not reenvio and marca in self._passou:
            self.estat["malha_laco"] = self.estat.get("malha_laco", 0) + 1
            return True  # esta mesma cópia já passou aqui: rodou em laço
        vizinhos = self._proximos(str(cab.get("d")), prio, evitar=de, redundante=False)
        if not vizinhos:
            return False  # guarda (na entrada persistente) até aparecer rota
        resta = None if expira is None else expira - self.parede()
        novo_cab = {**cab, "s": saltos + 1}
        quadro = _codifica(novo_cab, corpo)
        feito = False
        for viz in vizinhos:
            try:
                sid = self.no.enviar(viz, quadro, prio, CANAL_MALHA, ttl_s=resta)
            except FilaCheiaError:
                continue
            feito = True
            if cab.get("t") == "d" and prio < P4 and len(self._desp) < \
                    self.p.lembra_encaminhados:
                self._desp[sid] = _Despacho(bytes.fromhex(str(cab["id"])), str(cab["d"]), viz,
                                            prio, novo_cab, corpo, expira, False)
        if feito:
            self.estat["malha_encaminhadas"] += 1
            self._passou[marca] = None
            while len(self._passou) > 20000:
                self._passou.popitem(last=False)
        return feito

    def _trata_slot(self, cab: dict, corpo: bytes, c: Chegada, de: str) -> None:
        dest, origem, slot = str(cab.get("d")), str(cab.get("o")), str(cab.get("sl"))
        if dest == self.nome:
            sq = int(cab.get("sq", 0))
            if sq <= self._slot_seq_rx.get((origem, slot), 0):
                self.estat["slot_velho_rx"] += 1
                return
            self._slot_seq_rx[(origem, slot)] = sq
            self._slots_rx[(origem, slot)] = Chegada(0, c.id, origem, c.prio, int(cab.get(
                "c", 0)), corpo, c.recebido, c.expira)
            self.novidade.notify_all()
            return
        if int(cab.get("s", 1)) >= MAX_SALTOS or c.expira is None:
            return
        resta = c.expira - self.parede()
        if resta <= 0:
            return
        quadro = _codifica({**cab, "s": int(cab.get("s", 1)) + 1}, corpo)
        for viz in self._proximos(dest, c.prio, evitar=de, redundante=False):
            self.no.enviar_ultimo(viz, f"{origem}>{dest}:{slot}", quadro, resta, prio=c.prio,
                                  canal=CANAL_MALHA)

    # -- confirmações e reenvios ------------------------------------------------------

    def _manda_acks(self) -> None:
        for origem, ids in list(self._acks.items()):
            if origem not in self.rotas:
                continue  # sem rota: confirma quando voltar (a origem reenvia e a gente
                # confirma a repetida de novo)
            del self._acks[origem]
            for i in range(0, len(ids), 60):
                lote = b"".join(ids[i:i + 60])
                cab = {"t": "a", "o": self.nome, "d": origem, "s": 1}
                if origem in self._e2e_chaves:
                    cab["h"] = _assina_ack(self._e2e_chaves[origem], lote)
                quadro = _codifica(cab, lote)
                for viz in self._proximos(origem, P1):
                    try:
                        self.no.enviar(viz, quadro, P1, CANAL_MALHA, ttl_s=30.0)
                    except FilaCheiaError:
                        continue
                self.estat["malha_acks"] += 1
        for origem in list(self._acks):  # sem rota há tempo: não acumula sem fim
            self._acks[origem] = self._acks[origem][-2000:]

    def _confirmado(self, e2e: bytes) -> None:
        self._db.execute("DELETE FROM malha_saida WHERE id = ?", (e2e,))
        for sid, d in list(self._desp.items()):
            if d.e2e == e2e:
                if d.origem_aqui:
                    self.no.cancelar(d.vizinho, sid)  # a cópia redundante ainda na fila
                del self._desp[sid]

    def _salto_entregue(self, vizinho: str, sid: bytes) -> None:
        d = self._desp.pop(sid, None)
        if d is None:
            linha = self._db.execute("SELECT id, dest FROM malha_saida WHERE salto_id = ?",
                                     (sid,)).fetchone()
            if linha is None:
                return
            e2e, dest, origem_aqui = linha[0], linha[1], True
        else:
            e2e, dest, origem_aqui = d.e2e, d.dest, d.origem_aqui
        if not origem_aqui:
            return
        if vizinho == dest:  # entregue direto ao destino: está guardado lá, acabou
            self._confirmado(e2e)
        else:
            self._db.execute("UPDATE malha_saida SET salto_ok = ? WHERE id = ? AND "
                             "salto_ok IS NULL", (self.parede(), e2e))

    def _reenvios(self, agora: float) -> None:
        parede = self.parede()
        vencidas = self._db.execute(
            "DELETE FROM malha_saida WHERE expira IS NOT NULL AND expira <= ?",
            (parede,)).rowcount
        self.estat["malha_vencidas"] += vencidas
        em_fila = None
        # 1) cópias em fila de um vizinho que não é mais o caminho (morreu ou trocou):
        #    tira de lá e manda pelo caminho atual — a repetidora também faz isso
        for sid, d in list(self._desp.items()):
            if d.expira is not None and d.expira <= parede:
                del self._desp[sid]
                continue
            r = self.rotas.get(d.dest)
            vivo = self.no.vivo(d.vizinho)
            if r is None or (vivo and d.vizinho in (r.prox, self.segundas.get(
                    d.dest, r).prox)):
                continue
            if not self.no.cancelar(d.vizinho, sid):
                continue  # já saiu inteiro para o vizinho: a confirmação fim a fim decide
            del self._desp[sid]
            self.estat["malha_reenvios"] += 1
            if d.origem_aqui:
                linha = self._db.execute(
                    "SELECT dest, prio, canal, corpo, expira FROM malha_saida WHERE id = ?",
                    (d.e2e,)).fetchone()
                if linha is not None:
                    self._despacha(d.e2e, *linha)
            elif d.corpo is not None:
                self._encaminha({**d.canal_cab, "s": int(d.canal_cab["s"]) - 1}, d.corpo,
                                d.prio, d.expira, d.vizinho, reenvio=True)
        # 2) origem: sem confirmação fim a fim no prazo, ou sem cópia em salto nenhum
        #    (nunca despachada por falta de rota, ou de antes de um reinício)
        linhas = self._db.execute(
            "SELECT id, dest, prio, canal, corpo, expira, salto_id, salto_ok, tamanho "
            "FROM malha_saida ORDER BY prio, ordem LIMIT 2000").fetchall()
        meus = {d.e2e for d in self._desp.values() if d.origem_aqui}
        for e2e, dest, prio, canal, corpo, expira, salto_id, salto_ok, tam in linhas:
            if dest not in self.rotas:
                continue
            if salto_ok is None:
                if e2e in meus:
                    continue  # na fila do 1º salto (o passo 1 cuida se ele morrer)
                if em_fila is None:
                    em_fila = self.no.fila.ids_pendentes()
                if salto_id is not None and salto_id in em_fila:
                    continue
                self._despacha(e2e, dest, prio, canal, corpo, expira)
                continue
            r = self.rotas[dest]
            prazo = self.p.e2e_base_s.get(prio, 15.0) + 4 * r.custo * max(1.0, tam / 1200)
            if parede - salto_ok > prazo:
                self.estat["malha_reenvios"] += 1
                self._despacha(e2e, dest, prio, canal, corpo, expira)
