"""HTTP sobre o enlace: o Rover fala com a plataforma **através da base** sem mudar o protocolo.

O :class:`~valtexa_rover.runtime.client.DeviceClient` assina cada requisição com a chave
Ed25519 do Rover e entrega à HAL de rede (``Network.send``). :class:`RedeViaBase` é essa
HAL: embrulha método, caminho, cabeçalhos (com a assinatura) e corpo numa mensagem do
enlace, com a prioridade da rota, e espera a resposta que a base trouxe da nuvem. A base
repassa os bytes como vieram: **não consegue forjar** (não tem a chave do Rover; mudar um
byte invalida a assinatura) e a resposta com comando continua assinada pela plataforma.

Prioridade por rota (:func:`classifica`):

* P1 — comandos, teleop (slot "só o último vale"), missão, ``ack``/``complete``,
  resultado de comando, sessão e hora;
* P2 — telemetria (``/sync``, ``/telemetry``, ``/events``), ensaio de rede;
* P3 — resultados e eventos (``/edge/records``), abertura/conclusão de upload e pedaços de
  arquivos que **não** são originais;
* P4 — pedaços de originais (o Rover diz o ``tier`` na abertura do upload) e pacote de OTA.

Sem enlace vivo, ou sem resposta no prazo da classe, ``send`` levanta
:class:`~valtexa_rover.hal.interfaces.NetworkUnavailableError` — exatamente o que a
sincronização já trata: o item fica na fila do Rover, sem gastar tentativa. A base sem
internet responde ``503`` com ``X-Valtexa-Base: sem-internet``; isso também vira "sem rede".
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import secrets
import struct
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from valtexa_rover.enlace.no import P1, P2, P3, P4, NoDeEnlace
from valtexa_rover.hal.interfaces import DeviceInfo, HttpResponse, NetworkUnavailableError

CANAL_PEDIDO = 1
CANAL_RESPOSTA = 2
CANAL_CONTROLE = 3
CANAL_ARQUIVO = 4

SLOT_TELEOP = "teleop"
CABECALHO_BASE = "X-Valtexa-Base"
SEM_INTERNET = "sem-internet"

PRAZOS_PADRAO = {0: 5.0, 1: 4.0, 2: 15.0, 3: 30.0, 4: 120.0}
PRAZO_TELEOP = 1.0
PRAZO_VIDEO = 3.0
"""Quadro ao vivo mais velho que isto não serve ao operador: cai no caminho."""
SLOT_VIDEO = "video:"
_VIDEO = re.compile(r"^/v1/device/video/([^/]+)$")

_UPLOAD_CHUNK = re.compile(r"^/v1/device/uploads/([^/]+)/chunks/\d+$")


def codifica(cabecalho: dict, corpo: bytes = b"") -> bytes:
    j = json.dumps(cabecalho, separators=(",", ":"), ensure_ascii=False).encode()
    return struct.pack("!I", len(j)) + j + corpo


def decodifica(dados: bytes) -> tuple[dict, bytes]:
    (n,) = struct.unpack_from("!I", dados)
    return json.loads(dados[4:4 + n]), dados[4 + n:]


@dataclass
class Classificador:
    """Prioridade de cada rota; lembra o ``tier`` de cada upload aberto."""

    tiers: OrderedDict[str, str] = field(default_factory=OrderedDict)

    def __call__(self, metodo: str, caminho: str, corpo: bytes) -> tuple[int, str | None]:
        c = caminho.partition("?")[0]
        if c == "/v1/device/teleop":
            return P1, SLOT_TELEOP
        m = _VIDEO.match(c)
        if m:  # vídeo ao vivo: só o último quadro de cada câmera, abaixo de comando
            return P3, SLOT_VIDEO + m.group(1)
        if c == "/v1/device/uploads" and metodo == "POST":
            try:
                dados = json.loads(corpo)
                self.tiers[dados["upload_id"]] = dados.get("tier") or ""
                while len(self.tiers) > 4096:
                    self.tiers.popitem(last=False)
            except (ValueError, KeyError, TypeError):
                pass
            return P3, None
        m = _UPLOAD_CHUNK.match(c)
        if m:
            return (P4 if self.tiers.get(m.group(1)) == "original" else P3), None
        if c.startswith("/v1/device/ota"):
            return P4, None
        if c in ("/v1/device/sync", "/v1/device/telemetry", "/v1/device/events") or \
                c.startswith("/v1/device/ensaio/"):
            return P2, None
        if c.startswith("/v1/device/edge/") or c.startswith("/v1/device/uploads"):
            return P3, None
        return P1, None  # comandos, missão, sessão, hora: o que destrava o robô


classifica = Classificador()


class ClienteRpc:
    """Pedido → resposta sobre o enlace. Várias threads podem esperar ao mesmo tempo."""

    def __init__(self, no: NoDeEnlace, par: str, prazos: dict[int, float] | None = None
                 ) -> None:
        self.no = no
        self.par = par
        self.prazos = {**PRAZOS_PADRAO, **(prazos or {})}
        self._respostas: OrderedDict[str, tuple[dict, bytes]] = OrderedDict()
        self.atrasadas = 0

    def requisitar(self, metodo: str, caminho: str, corpo: bytes, cabecalhos: dict[str, str],
                   prio: int, *, slot: str | None = None, prazo_s: float | None = None
                   ) -> tuple[int, bytes, dict[str, str]]:
        prazo = prazo_s if prazo_s is not None else (
            PRAZO_TELEOP if slot == SLOT_TELEOP
            else PRAZO_VIDEO if slot and slot.startswith(SLOT_VIDEO) else self.prazos[prio])
        rid = secrets.token_hex(8)
        limite = self.no.relogio.parede() + prazo
        msg = codifica({"rid": rid, "m": metodo, "p": caminho, "h": cabecalhos,
                        "prazo": limite, "slot": slot}, corpo)
        if slot is not None:
            self.no.enviar_ultimo(self.par, slot, msg, prazo, prio=prio)
        else:
            self.no.enviar(self.par, msg, prio, CANAL_PEDIDO, ttl_s=prazo)
        fim = self.no.relogio.agora() + prazo
        with self.no.novidade:
            while True:
                self._colhe(slot)
                if rid in self._respostas:
                    cab, dados = self._respostas.pop(rid)
                    return int(cab["s"]), dados, dict(cab.get("h") or {})
                resta = fim - self.no.relogio.agora()
                if resta <= 0:
                    raise NetworkUnavailableError(
                        f"sem resposta da base em {prazo:.1f}s ({metodo} {caminho})")
                self.no.novidade.wait(min(resta, 0.05))

    def _colhe(self, slot: str | None) -> None:
        for c in self.no.chegadas(CANAL_RESPOSTA, 200):
            self.no.consumida(c.seq)
            try:
                cab, dados = decodifica(c.corpo)
            except (ValueError, struct.error):
                continue
            self._guarda(cab, dados)
        if slot is not None:
            c = self.no.ultimo(self.par, slot)
            if c is not None:
                with contextlib.suppress(ValueError, struct.error):
                    self._guarda(*decodifica(c.corpo))

    def _guarda(self, cab: dict, dados: bytes) -> None:
        self._respostas[cab.get("rid", "")] = (cab, dados)
        while len(self._respostas) > 256:  # resposta de quem já desistiu
            self._respostas.popitem(last=False)
            self.atrasadas += 1


RECIBO_ALGORITMO = "VALTEXA-ED25519-RECIBO-V1"
RECIBO_PEDIDO = "X-Valtexa-Recibo"
RECIBO_CABECALHO = "x-valtexa-recibo-assinatura"


def texto_do_recibo(nonce: str, status: int, corpo: bytes) -> bytes:
    """O mesmo texto que a plataforma assina (``saas/valtexa/api/middleware.py``)."""
    return (f"{RECIBO_ALGORITMO}\n{nonce}\n{int(status)}\n"
            f"{hashlib.sha256(corpo).hexdigest()}").encode()


def confere_recibo(chave_plataforma: str, nonce: str, status: int, corpo: bytes,
                   cabecalhos: dict[str, str]) -> bool:
    assinatura = next((v for k, v in cabecalhos.items() if k.lower() == RECIBO_CABECALHO), None)
    if not assinatura or not nonce:
        return False
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(chave_plataforma)).verify(
            bytes.fromhex(assinatura), texto_do_recibo(nonce, status, corpo))
    except (InvalidSignature, ValueError):
        return False
    return True


class RedeViaBase:
    """HAL de rede do Rover quando a nuvem está do outro lado da base (rádio).

    **Recibo assinado** (Parte 13): a base repassa os bytes, mas um "200 aceito" vindo dela
    faria o Rover apagar da fila de sincronização o que talvez nunca chegou à plataforma
    (base com defeito ou adulterada). Por isso todo pedido que muda estado (não ``GET``) vai
    com ``X-Valtexa-Recibo: 1`` e a resposta 2xx só vale com a assinatura da plataforma
    sobre ``nonce do pedido | status | sha256 do corpo``; sem ela, é "sem rede" (o dado
    fica no Rover). ``chave_plataforma`` vem da identidade do Rover (o ``RoverCore`` liga);
    ``exige_recibo = False`` só para falar com uma plataforma anterior à Parte 13.
    """

    def __init__(self, cliente: ClienteRpc, classificador: Classificador | None = None, *,
                 chave_plataforma: str | None = None, exige_recibo: bool = True) -> None:
        self.cliente = cliente
        self.classificador = classificador or Classificador()
        self._sem_nuvem_em = -1e9
        self.chave_plataforma = chave_plataforma
        self.exige_recibo = exige_recibo
        self.recibos_recusados = 0
        self.recibos_conferidos = 0

    def is_online(self) -> bool:
        """A plataforma é alcançável: algum rádio vivo **e** a base com internet.

        Com a Starlink caída a sincronização fica parada (um ``ROVER_OFFLINE``, não um par
        ONLINE/OFFLINE a cada passo) e volta sozinha quando o batimento da base disser
        ``"nuvem": true``. Comando e teleop continuam tentando (a base pode servir missão
        do cache e o enlace continua vivo para o PARAR).
        """
        return self.cliente.no.vivo(self.cliente.par) and self.nuvem_ok()

    def nuvem_ok(self) -> bool:
        """A base disse no batimento que tem internet (e não respondeu "sem internet" há
        menos de 1 s — o batimento seguinte confirma)."""
        info = self.cliente.no.info_do_par(self.cliente.par)
        recente = self.cliente.no.relogio.agora() - self._sem_nuvem_em < 1.0
        return bool(info.get("nuvem", True)) and not recente

    def send(self, method: str, path: str, body: bytes, headers: dict[str, str]) -> HttpResponse:
        if not self.cliente.no.vivo(self.cliente.par):
            raise NetworkUnavailableError("nenhum enlace de rádio vivo com a base")
        prio, slot = self.classificador(method, path, body)
        muda = method.upper() not in ("GET", "HEAD")
        recibo = muda and self.exige_recibo and bool(self.chave_plataforma)
        if recibo:  # fora da assinatura do pedido (só método, caminho, hora, nonce e corpo)
            headers = {**headers, RECIBO_PEDIDO: "1"}
        status, dados, cab = self.cliente.requisitar(method, path, body, headers, prio,
                                                     slot=slot)
        if status == 503 and cab.get(CABECALHO_BASE) == SEM_INTERNET:
            self._sem_nuvem_em = self.cliente.no.relogio.agora()
            raise NetworkUnavailableError("a base está sem internet (Starlink)")
        if recibo and 200 <= status < 300:
            nonce = next((v for k, v in headers.items() if k.lower() == "x-valtexa-nonce"), "")
            if not confere_recibo(self.chave_plataforma, nonce, status, dados, cab):
                self.recibos_recusados += 1
                raise NetworkUnavailableError(
                    f"resposta {status} sem recibo válido da plataforma ({method} {path}): "
                    "o dado fica no Rover")
            self.recibos_conferidos += 1
        return HttpResponse(status, dados, cab)

    def describe(self) -> DeviceInfo:
        return DeviceInfo("rede", "via base (enlace de rádio)", False,
                          {"saude": self.cliente.no.saude()["pares"].get(self.cliente.par)})


def _com_expira_local(cab: dict, c) -> dict:
    """Validade do pedido **no relógio de quem recebeu**.

    O enlace leva a validade como tempo restante (``ttl_ms`` em cada pedaço) e o receptor a
    converte para o próprio relógio (``Chegada.expira``). O ``prazo`` do cabeçalho é a hora
    de parede do Rover — com o relógio do Rover errado (sem GNSS ainda, RTC atrasado), a
    base recusaria tudo (Rover atrasado) ou aceitaria pedido morto (adiantado). Fica só
    como informação.
    """
    cab = dict(cab)
    if c.expira is not None:
        cab["_expira"] = c.expira
    return cab


def resta_s(no, cab: dict) -> float:
    """Segundos de validade que sobram do pedido, no relógio deste nó."""
    expira = cab.get("_expira")
    if expira is None:  # pedido sem validade pelo enlace (versão antiga): o do cabeçalho
        expira = float(cab.get("prazo", 0))
    return float(expira) - no.parede()


def atende(no: NoDeEnlace, tratador, *, limite: int = 64) -> int:
    """Lado da base: entrega os pedidos que chegaram a ``tratador(par, cab, corpo, prio)``.

    ``tratador`` devolve ``(status, corpo, cabeçalhos)`` ou ``None`` (vai responder depois,
    com :func:`responde`). Pedido vencido não é repassado. Devolve quantos atendeu.
    """
    n = 0
    for c in no.chegadas(CANAL_PEDIDO, limite):
        no.consumida(c.seq)
        try:
            cab, corpo = decodifica(c.corpo)
        except (ValueError, struct.error):
            continue
        cab = _com_expira_local(cab, c)
        if resta_s(no, cab) <= 0:
            no.estat["vencidas_rx"] += 1
            continue
        n += 1
        r = tratador(c.par, cab, corpo, c.prio)
        if r is not None:
            responde(no, c.par, cab, c.prio, *r)
    for par in no.pares():
        # teleop e cada câmera do vídeo ao vivo: "só o último vale"
        for slot in no.slots_recebidos(par):
            c = no.ultimo(par, slot)
            if c is None:
                continue
            try:
                cab, corpo = decodifica(c.corpo)
            except (ValueError, struct.error):
                continue
            cab = _com_expira_local(cab, c)
            if resta_s(no, cab) <= 0:
                continue
            n += 1
            r = tratador(par, cab, corpo, c.prio)
            if r is not None:
                responde(no, par, cab, c.prio, *r)
    return n


def responde(no: NoDeEnlace, par: str, cab: dict, prio: int, status: int, corpo: bytes,
             cabecalhos: dict[str, Any] | None = None) -> bool:
    """Manda a resposta de volta, com a validade que sobrou do pedido. ``False`` se venceu."""
    resta = resta_s(no, cab)
    if resta <= 0:
        return False
    msg = codifica({"rid": cab["rid"], "s": status, "h": dict(cabecalhos or {})}, corpo)
    if cab.get("slot"):
        no.enviar_ultimo(par, cab["slot"], msg, resta, prio=prio, canal=CANAL_RESPOSTA)
    else:
        no.enviar(par, msg, prio, CANAL_RESPOSTA, ttl_s=resta)
    return True


class Executor:
    """Roda ``no.passo()`` numa thread, em tempo real, esperando os sockets com ``select``."""

    def __init__(self, no: NoDeEnlace, intervalo_s: float = 0.005,
                 ao_passo=None) -> None:
        self.no = no
        self.intervalo_s = intervalo_s
        self.ao_passo = ao_passo
        self._parar = threading.Event()
        self._thread: threading.Thread | None = None
        self.erros = 0

    def iniciar(self) -> Executor:
        self._thread = threading.Thread(target=self._laco, daemon=True,
                                        name=f"enlace-{self.no.nome}")
        self._thread.start()
        return self

    def _laco(self) -> None:
        import select

        fds = [f for e in self.no.enlaces.values() if (f := e.transporte.fileno()) is not None]
        while not self._parar.is_set():
            if fds:
                try:
                    select.select(fds, [], [], self.intervalo_s)
                except (OSError, ValueError):
                    self._parar.wait(self.intervalo_s)
            else:
                self._parar.wait(self.intervalo_s)
            try:
                self.no.passo()
                if self.ao_passo is not None:
                    self.ao_passo()
            except Exception:  # o laço do enlace não pode morrer por um pacote ruim
                self.erros += 1
                if self.erros > 1000:
                    raise

    def parar(self) -> None:
        self._parar.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
