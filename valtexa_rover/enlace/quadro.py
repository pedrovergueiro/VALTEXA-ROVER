"""Formato dos pacotes no fio e dos registros dentro deles.

Pacote (um datagrama UDP; cabe na MTU do enlace)::

    "VX" | versão (1) | tipo (1) | sessão (8) | contador (8) | corpo ... | CRC32 (4)

* **CRC32** sobre tudo o que vem antes: corrupção de bit é descartada antes de gastar
  criptografia (e conta como perda na saúde do enlace).
* ``tipo``: ``HS1``/``HS2`` (aperto de mão, :mod:`.cripto`), ``DADOS`` (corpo cifrado e
  autenticado com ChaCha20-Poly1305; o cabeçalho de 20 bytes é o dado associado) e
  ``REINICIAR`` (aviso **não autenticado** de "não conheço esta sessão": no máximo faz o
  iniciador refazer o aperto de mão, com limite de taxa).
* ``contador`` é o número de sequência do pacote na sessão e no sentido; é também o nonce
  da cifra (nunca se repete numa sessão; cada aperto de mão cria chaves novas).

Dentro do corpo decifrado vão um ou mais **registros** (TLV), para o ACK pegar carona no
dado:

* ``FRAG`` — um pedaço de mensagem: id (16), prioridade, canal, flags, índice, total,
  validade restante (ms), [slot e sequência do slot], dados;
* ``ACK`` — ACK seletivo: para cada mensagem, o mapa de bits dos pedaços recebidos
  (janela de 64 a partir de ``base``) ou "entregue inteira";
* ``BAT`` — batimento: carimbo de envio, eco do último batimento do outro lado (RTT),
  quantos pacotes recebi por este enlace (perda) e um JSON curto de estado.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field

MAGIA = b"VX"
VERSAO = 1

HS1, HS2, DADOS, REINICIAR = 1, 2, 3, 4
TIPOS = (HS1, HS2, DADOS, REINICIAR)

CABECALHO = struct.Struct("!2sBB8sQ")  # 20 bytes
TAM_CABECALHO = CABECALHO.size
TAM_CRC = 4
TAM_TAG = 16  # Poly1305

# Registros
R_FRAG, R_ACK, R_BAT = 1, 2, 3

F_SLOT = 0x01  # "só o último vale" (teleop)

_FRAG = struct.Struct("!16sBBBHHI")  # id, prio, canal, flags, idx, total, ttl_ms
_SLOT = struct.Struct("!BQ")  # tamanho do nome do slot, sequência do slot
_ACK_ITEM = struct.Struct("!16sHQ")  # id, base, mapa de 64 bits
ACK_ENTREGUE = 0xFFFF  # base especial: mensagem entregue inteira
_BAT = struct.Struct("!IdIdId")  # seq, t_envio, eco_seq, eco_t, recebidos, retido

SEM_TTL = 0xFFFFFFFF


class QuadroInvalidoError(ValueError):
    """Pacote corrompido, truncado ou de outra versão: descartado e contado."""


@dataclass(frozen=True, slots=True)
class Cabecalho:
    tipo: int
    sessao: bytes
    contador: int

    def bytes(self) -> bytes:
        return CABECALHO.pack(MAGIA, VERSAO, self.tipo, self.sessao, self.contador)


def monta_pacote(cab: Cabecalho, corpo: bytes) -> bytes:
    dados = cab.bytes() + corpo
    return dados + struct.pack("!I", zlib.crc32(dados))


def abre_pacote(pacote: bytes) -> tuple[Cabecalho, bytes, bytes]:
    """``(cabeçalho, bytes do cabeçalho, corpo)``; :class:`QuadroInvalidoError` se não confere."""
    if len(pacote) < TAM_CABECALHO + TAM_CRC:
        raise QuadroInvalidoError("pacote curto")
    dados, crc = pacote[:-TAM_CRC], struct.unpack("!I", pacote[-TAM_CRC:])[0]
    if zlib.crc32(dados) != crc:
        raise QuadroInvalidoError("CRC não confere")
    magia, versao, tipo, sessao, contador = CABECALHO.unpack_from(dados)
    if magia != MAGIA or versao != VERSAO or tipo not in TIPOS:
        raise QuadroInvalidoError("cabeçalho desconhecido")
    return Cabecalho(tipo, sessao, contador), dados[:TAM_CABECALHO], dados[TAM_CABECALHO:]


# ---------------------------------------------------------------------------
# Registros
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Frag:
    msg_id: bytes
    prio: int
    canal: int
    indice: int
    total: int
    dados: bytes
    ttl_ms: int | None = None
    slot: str | None = None
    slot_seq: int = 0

    def codifica(self) -> bytes:
        flags = F_SLOT if self.slot is not None else 0
        ttl = SEM_TTL if self.ttl_ms is None else max(0, min(SEM_TTL - 1, int(self.ttl_ms)))
        partes = [_FRAG.pack(self.msg_id, self.prio, self.canal, flags, self.indice,
                             self.total, ttl)]
        if self.slot is not None:
            nome = self.slot.encode()
            partes += [_SLOT.pack(len(nome), self.slot_seq), nome]
        partes.append(self.dados)
        return b"".join(partes)

    @staticmethod
    def sobrecarga(slot: str | None = None) -> int:
        extra = _SLOT.size + len(slot.encode()) if slot is not None else 0
        return 3 + _FRAG.size + extra  # 3 = tipo + tamanho do registro


@dataclass(slots=True)
class Ack:
    # (msg_id, base, mapa) — base == ACK_ENTREGUE: entregue inteira
    itens: list[tuple[bytes, int, int]] = field(default_factory=list)

    def codifica(self) -> bytes:
        return struct.pack("!H", len(self.itens)) + b"".join(
            _ACK_ITEM.pack(m, b, mapa) for m, b, mapa in self.itens)

    TAM_ITEM = _ACK_ITEM.size


@dataclass(slots=True)
class Bat:
    seq: int
    t_envio: float
    eco_seq: int
    eco_t: float
    recebidos: int
    retido: float  # quanto tempo o eco ficou parado aqui antes de sair (desconta do RTT)
    info: bytes = b""

    def codifica(self) -> bytes:
        return _BAT.pack(self.seq, self.t_envio, self.eco_seq, self.eco_t, self.recebidos,
                         self.retido) + self.info


def registro(tipo: int, conteudo: bytes) -> bytes:
    return struct.pack("!BH", tipo, len(conteudo)) + conteudo


def le_registros(corpo: bytes) -> list[Frag | Ack | Bat]:
    """Decodifica os registros de um corpo decifrado (desconhecidos são pulados)."""
    saida: list[Frag | Ack | Bat] = []
    pos = 0
    while pos + 3 <= len(corpo):
        tipo, tamanho = struct.unpack_from("!BH", corpo, pos)
        pos += 3
        conteudo = corpo[pos:pos + tamanho]
        if len(conteudo) != tamanho:
            raise QuadroInvalidoError("registro truncado")
        pos += tamanho
        if tipo == R_FRAG:
            saida.append(_le_frag(conteudo))
        elif tipo == R_ACK:
            (n,) = struct.unpack_from("!H", conteudo)
            itens = [_ACK_ITEM.unpack_from(conteudo, 2 + i * _ACK_ITEM.size) for i in range(n)]
            saida.append(Ack(list(itens)))
        elif tipo == R_BAT:
            campos = _BAT.unpack_from(conteudo)
            saida.append(Bat(*campos, info=conteudo[_BAT.size:]))
    return saida


def _le_frag(c: bytes) -> Frag:
    msg_id, prio, canal, flags, indice, total, ttl = _FRAG.unpack_from(c)
    pos = _FRAG.size
    slot, slot_seq = None, 0
    if flags & F_SLOT:
        n, slot_seq = _SLOT.unpack_from(c, pos)
        pos += _SLOT.size
        slot = c[pos:pos + n].decode()
        pos += n
    if total == 0 or indice >= total or prio > 4:
        raise QuadroInvalidoError("fragmento incoerente")
    return Frag(msg_id, prio, canal, indice, total, c[pos:],
                None if ttl == SEM_TTL else ttl, slot, slot_seq)
