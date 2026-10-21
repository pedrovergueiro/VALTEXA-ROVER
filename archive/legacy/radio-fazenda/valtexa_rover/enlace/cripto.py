"""Criptografia autenticada **por salto** (Rover ⇄ base), com X25519 e ChaCha20-Poly1305.

Cada nó tem uma chave estática X25519 só do enlace (``enlace.key``, gerada no próprio nó,
permissão 0600, nunca no código nem no repositório). A pública vai para o outro lado
**assinada pela identidade Ed25519 provisionada** (:func:`publica_assinada`): a base só
aceita um Rover cuja chave de enlace foi assinada pela chave que a plataforma conhece.

Aperto de mão (dois pacotes, no estilo Noise KK — os dois lados já conhecem a estática
do outro):

* ``HS1`` (iniciador → respondedor): ``id(8) | e_i(32) | tag``; o ``tag`` sai de
  ``DH(e_i, s_r) ‖ DH(s_i, s_r)``: só quem tem a estática do iniciador o produz.
* ``HS2`` (respondedor → iniciador): ``e_r(32) | tag``, já com a chave da sessão.
* Chaves da sessão: HKDF-SHA256 sobre ``DH(e_i,e_r) ‖ DH(e_i,s_r) ‖ DH(s_i,e_r) ‖
  DH(s_i,s_r)`` → uma chave por sentido. Efêmeras novas a cada aperto de mão (sigilo
  futuro) e o contador recomeça com chave nova: **nonce nunca se repete** na mesma chave.
* **Anti-replay**: janela deslizante de 1024 contadores por sessão; pacote repetido ou
  velho demais é descartado. O respondedor só passa a *usar* uma sessão nova depois do
  primeiro pacote ``DADOS`` válido nela (um ``HS1`` repetido por um atacante não derruba
  a sessão em uso).

Isto protege o salto de rádio. **Não substitui** a assinatura Ed25519 fim a fim da
plataforma: a base decifra o salto, mas não tem a chave da plataforma (não forja comando)
nem a do Rover (não forja dado).
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import struct
from dataclasses import dataclass, field
from pathlib import Path

from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from valtexa_rover.enlace.quadro import (
    DADOS,
    HS1,
    HS2,
    TAM_TAG,
    Cabecalho,
    monta_pacote,
)

JANELA_REPLAY = 1024
ALGORITMO_PUBLICA = "VALTEXA-ENLACE-X25519-V1"
_NONCE_HS = bytes(12)


class FalhaDeCriptoError(ValueError):
    """Tag inválida, par desconhecido, replay: o pacote é descartado e contado."""


def _pub(chave: X25519PrivateKey) -> bytes:
    return chave.public_key().public_bytes_raw()


def id_do_par(publica: bytes) -> bytes:
    """8 bytes que identificam a estática no ``HS1`` (sem mandar a pública inteira)."""
    return hashlib.sha256(publica).digest()[:8]


class ChaveDeEnlace:
    """A chave estática X25519 do nó. ``repr`` não mostra o segredo."""

    def __init__(self, privada: X25519PrivateKey) -> None:
        self._privada = privada
        self.publica = _pub(privada)
        self.id = id_do_par(self.publica)

    def __repr__(self) -> str:
        return f"ChaveDeEnlace(publica={self.publica.hex()[:16]}…)"

    @classmethod
    def gerar(cls) -> ChaveDeEnlace:
        return cls(X25519PrivateKey.generate())

    @classmethod
    def carregar_ou_criar(cls, caminho: Path | str) -> ChaveDeEnlace:
        """Lê ``caminho``; se não existe, gera **aqui** e grava com 0600."""
        caminho = Path(caminho)
        if caminho.exists():
            bruto = bytes.fromhex(caminho.read_text(encoding="utf-8").strip())
            return cls(X25519PrivateKey.from_private_bytes(bruto))
        chave = cls.gerar()
        caminho.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(caminho, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(chave._privada.private_bytes_raw().hex())
        return chave

    def dh(self, publica: bytes) -> bytes:
        return self._privada.exchange(X25519PublicKey.from_public_bytes(publica))


def publica_assinada(chave: ChaveDeEnlace, nome: str, privada_ed25519_hex: str) -> dict:
    """Documento ``{nome, enlace_pub, assinatura}`` assinado pela identidade do nó."""
    texto = f"{ALGORITMO_PUBLICA}\n{nome}\n{chave.publica.hex()}".encode()
    assinante = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(privada_ed25519_hex))
    return {"nome": nome, "enlace_pub": chave.publica.hex(),
            "assinatura": assinante.sign(texto).hex()}


def confere_publica_assinada(doc: dict, publica_ed25519_hex: str) -> bytes:
    """Devolve a pública X25519 se a assinatura confere com a identidade conhecida."""
    texto = f"{ALGORITMO_PUBLICA}\n{doc['nome']}\n{doc['enlace_pub']}".encode()
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(publica_ed25519_hex)).verify(
            bytes.fromhex(doc["assinatura"]), texto)
    except (InvalidSignature, ValueError) as exc:
        raise FalhaDeCriptoError(f"chave de enlace de {doc.get('nome')!r} sem assinatura válida") \
            from exc
    return bytes.fromhex(doc["enlace_pub"])


def carrega_pares(caminho: Path | str) -> dict[str, bytes]:
    """``pares.json``: ``{"nome": "<pública hex>"}`` (ou o documento assinado já conferido)."""
    dados = json.loads(Path(caminho).read_text(encoding="utf-8"))
    return {nome: bytes.fromhex(v if isinstance(v, str) else v["enlace_pub"])
            for nome, v in dados.items()}


def _hkdf(ikm: bytes, tamanho: int, salt: bytes, info: bytes) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=tamanho, salt=salt, info=info).derive(ikm)


class JanelaAntiReplay:
    """Aceita cada contador uma vez; recusa os mais velhos que a janela."""

    def __init__(self, tamanho: int = JANELA_REPLAY) -> None:
        self.tamanho = tamanho
        self.maior = 0
        self.mapa = 0  # bit i = contador (maior - i) já visto

    def confere(self, ctr: int) -> bool:
        if ctr == 0:
            return False
        if ctr > self.maior:
            return True
        atras = self.maior - ctr
        return atras < self.tamanho and not (self.mapa >> atras) & 1

    def marca(self, ctr: int) -> None:
        if ctr > self.maior:
            desloca = ctr - self.maior
            self.mapa = ((self.mapa << desloca) | 1) & ((1 << self.tamanho) - 1)
            self.maior = ctr
        else:
            self.mapa |= 1 << (self.maior - ctr)


@dataclass
class Sessao:
    sid: bytes
    iniciador: bool
    k_envio: bytes
    k_recebe: bytes
    criada: float
    confirmada: bool = False
    ctr_envio: int = 0
    janela: JanelaAntiReplay = field(default_factory=JanelaAntiReplay)
    ultimo_rx: float = 0.0

    def __post_init__(self) -> None:
        self._env = ChaCha20Poly1305(self.k_envio)
        self._rec = ChaCha20Poly1305(self.k_recebe)
        self._dir_env = b"\x00\x00\x00\x00" if self.iniciador else b"\x00\x00\x00\x01"
        self._dir_rec = b"\x00\x00\x00\x01" if self.iniciador else b"\x00\x00\x00\x00"

    def sela(self, corpo: bytes) -> bytes:
        self.ctr_envio += 1
        cab = Cabecalho(DADOS, self.sid, self.ctr_envio)
        ad = cab.bytes()
        cifrado = self._env.encrypt(self._dir_env + struct.pack("!Q", self.ctr_envio), corpo, ad)
        return monta_pacote(cab, cifrado)

    def abre(self, cab: Cabecalho, ad: bytes, corpo: bytes, agora: float) -> bytes:
        if not self.janela.confere(cab.contador):
            raise FalhaDeCriptoError("replay ou fora da janela")
        try:
            claro = self._rec.decrypt(self._dir_rec + struct.pack("!Q", cab.contador), corpo, ad)
        except InvalidTag as exc:
            raise FalhaDeCriptoError("tag inválida") from exc
        self.janela.marca(cab.contador)
        self.confirmada = True
        self.ultimo_rx = agora
        return claro


def _chaves_da_sessao(dhs: bytes, e_i: bytes, e_r: bytes, s_i: bytes, s_r: bytes
                      ) -> tuple[bytes, bytes]:
    okm = _hkdf(dhs, 64, e_i + e_r, b"vx-sessao-v1" + s_i + s_r)
    return okm[:32], okm[32:]  # (iniciador→respondedor, respondedor→iniciador)


@dataclass
class ApertoPendente:
    sid: bytes
    efemera: X25519PrivateKey
    par_publica: bytes
    pacote: bytes
    enviado_em: float
    tentativas: int = 0


def inicia_aperto(chave: ChaveDeEnlace, par_publica: bytes, agora: float) -> ApertoPendente:
    sid = secrets.token_bytes(8)
    e = X25519PrivateKey.generate()
    e_pub = _pub(e)
    k1 = _hkdf(e.exchange(X25519PublicKey.from_public_bytes(par_publica)) + chave.dh(par_publica),
               32, b"vx-hs1", e_pub + par_publica)
    cab = Cabecalho(HS1, sid, 0)
    corpo_sem_tag = chave.id + e_pub
    tag = ChaCha20Poly1305(k1).encrypt(_NONCE_HS, b"", cab.bytes() + corpo_sem_tag)
    return ApertoPendente(sid, e, par_publica, monta_pacote(cab, corpo_sem_tag + tag), agora)


def responde_aperto(chave: ChaveDeEnlace, pares: dict[bytes, bytes], cab: Cabecalho,
                    ad: bytes, corpo: bytes, agora: float) -> tuple[bytes, Sessao, bytes]:
    """``(publica do par, sessão nova não confirmada, pacote HS2)``."""
    if len(corpo) != 8 + 32 + TAM_TAG:
        raise FalhaDeCriptoError("HS1 com tamanho errado")
    id_i, e_i, tag = corpo[:8], corpo[8:40], corpo[40:]
    s_i = pares.get(id_i)
    if s_i is None:
        raise FalhaDeCriptoError("par desconhecido")
    k1 = _hkdf(chave.dh(e_i) + chave.dh(s_i), 32, b"vx-hs1", e_i + chave.publica)
    try:
        ChaCha20Poly1305(k1).decrypt(_NONCE_HS, tag, ad + id_i + e_i)
    except InvalidTag as exc:
        raise FalhaDeCriptoError("HS1 com tag inválida") from exc
    e = X25519PrivateKey.generate()
    e_r = _pub(e)
    dhs = (e.exchange(X25519PublicKey.from_public_bytes(e_i)) + chave.dh(e_i)
           + e.exchange(X25519PublicKey.from_public_bytes(s_i)) + chave.dh(s_i))
    k_ir, k_ri = _chaves_da_sessao(dhs, e_i, e_r, s_i, chave.publica)
    sessao = Sessao(cab.sessao, False, k_envio=k_ri, k_recebe=k_ir, criada=agora)
    cab2 = Cabecalho(HS2, cab.sessao, 0)
    tag2 = ChaCha20Poly1305(k_ri).encrypt(_NONCE_HS, b"", cab2.bytes() + e_r)
    return s_i, sessao, monta_pacote(cab2, e_r + tag2)


def conclui_aperto(chave: ChaveDeEnlace, pend: ApertoPendente, ad: bytes, corpo: bytes,
                   agora: float) -> Sessao:
    if len(corpo) != 32 + TAM_TAG:
        raise FalhaDeCriptoError("HS2 com tamanho errado")
    e_r, tag = corpo[:32], corpo[32:]
    e_i = _pub(pend.efemera)
    s_r = pend.par_publica
    dhs = (pend.efemera.exchange(X25519PublicKey.from_public_bytes(e_r))
           + pend.efemera.exchange(X25519PublicKey.from_public_bytes(s_r))
           + chave.dh(e_r) + chave.dh(s_r))
    k_ir, k_ri = _chaves_da_sessao(dhs, e_i, e_r, chave.publica, s_r)
    try:
        ChaCha20Poly1305(k_ri).decrypt(_NONCE_HS, tag, ad + e_r)
    except InvalidTag as exc:
        raise FalhaDeCriptoError("HS2 com tag inválida") from exc
    return Sessao(pend.sid, True, k_envio=k_ir, k_recebe=k_ri, criada=agora, confirmada=True,
                  ultimo_rx=agora)
