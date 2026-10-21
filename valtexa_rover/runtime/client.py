"""Identidade do robô e cliente assinado do protocolo (``saas/docs/DEVICE_PROTOCOL.md``).

Toda requisição sai assinada com Ed25519 pela chave privada **do robô**, que
nunca sai dele (nem em log: ``repr`` da identidade esconde a chave). O texto
canônico é o da seção 4 do protocolo. O transporte é a HAL de rede — este
módulo não sabe se do outro lado há HTTPS, um rádio ou um teste.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import secrets
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from valtexa_rover.clock import Clock
from valtexa_rover.hal.interfaces import HttpResponse, Network, NetworkUnavailableError

REQUEST_ALGORITHM = "VALTEXA-ED25519-REQUEST-V1"


@dataclass(frozen=True)
class DeviceIdentity:
    """O que o provisionamento deixou no robô."""

    device_id: str
    key_id: str
    private_key: str
    platform_public_key: str | None = None
    ota_public_key: str | None = None
    """Confere os pacotes de atualização (OTA). Vem no provisionamento."""

    def __repr__(self) -> str:  # a chave privada nunca aparece em log
        return f"DeviceIdentity(device_id={self.device_id!r}, key_id={self.key_id!r})"

    @classmethod
    def load(cls, path: Path | str) -> DeviceIdentity:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            device_id=data["device_id"],
            key_id=data["key_id"],
            private_key=data["private_key"],
            platform_public_key=data.get("platform_public_key"),
            ota_public_key=data.get("ota_public_key"),
        )

    def save(self, path: Path | str) -> None:
        """Grava com permissão 0600: só o usuário do Runtime lê."""
        destino = Path(path)
        destino.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(destino, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "device_id": self.device_id,
                    "key_id": self.key_id,
                    "private_key": self.private_key,
                    "platform_public_key": self.platform_public_key,
                    "ota_public_key": self.ota_public_key,
                },
                fh,
            )


def generate_keypair() -> tuple[str, str]:
    """``(privada_hex, publica_hex)`` — gerado **no robô**, no primeiro boot."""
    chave = Ed25519PrivateKey.generate()
    return chave.private_bytes_raw().hex(), chave.public_key().public_bytes_raw().hex()


def provision(url: str, token: str, serial: str, destino: Path | str, *,
              abrir=urllib.request.urlopen) -> DeviceIdentity:
    """Primeiro boot: gera o par **aqui** e manda à plataforma só a chave pública.

    O token de provisionamento é de uso único; só vai por HTTPS (``http`` só para
    ``localhost``, em homologação). Uma identidade que já existe não é sobrescrita: o robô
    perderia a chave que a plataforma conhece.
    """
    destino = Path(destino)
    if destino.exists():
        raise FileExistsError(f"{destino} já existe: este robô já foi provisionado.")
    partes = urllib.parse.urlsplit(url)
    if partes.scheme != "https" and partes.hostname not in ("localhost", "127.0.0.1"):
        raise ValueError(f"O provisionamento só vai por HTTPS (recebi {url!r}).")
    privada, publica = generate_keypair()
    corpo = json.dumps({"provisioning_token": token, "serial_number": serial,
                        "public_key": publica}).encode()
    pedido = urllib.request.Request(url.rstrip("/") + "/v1/device/provision", data=corpo,
                                    method="POST", headers={"Content-Type": "application/json"})
    with abrir(pedido, timeout=30) as resposta:
        dados = json.loads(resposta.read())
    identidade = DeviceIdentity(dados["device_id"], dados["key_id"], privada,
                                dados["platform_public_key"], dados.get("ota_public_key"))
    identidade.save(destino)
    return identidade


def canonical_request(
    method: str, path: str, query: str, timestamp: str, nonce: str, body: bytes
) -> bytes:
    return "\n".join(
        [REQUEST_ALGORITHM, method.upper(), path, query or "", timestamp, nonce,
         hashlib.sha256(body).hexdigest()]
    ).encode("utf-8")


class DeviceClient:
    """Requisições assinadas sobre a HAL de rede."""

    def __init__(self, identity: DeviceIdentity, network: Network, clock: Clock) -> None:
        self.identity = identity
        self.network = network
        self.clock = clock
        self._key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(identity.private_key))
        self.contador = None
        """Gancho do orçamento de dados: ``contador(método, caminho, enviados, recebidos)``
        a cada requisição que trafegou (:meth:`OrcamentoDados.contar`)."""

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: object | None = None,
        raw: bytes | None = None,
        content_type: str | None = None,
        extra_headers: dict[str, str] | None = None,
        gzip_body: bool = False,
    ) -> HttpResponse:
        if json_body is not None:
            body = json.dumps(json_body, separators=(",", ":"), ensure_ascii=False).encode()
            content_type = "application/json"
        else:
            body = raw or b""
        if gzip_body:
            # A assinatura é dos bytes que trafegam (comprimidos): a plataforma confere antes
            # de descomprimir. ``mtime=0``: o mesmo corpo dá os mesmos bytes.
            body = gzip.compress(body, compresslevel=6, mtime=0)
            extra_headers = {**(extra_headers or {}), "Content-Encoding": "gzip"}
        caminho, _, query = path.partition("?")
        timestamp = str(int(self.clock.now().timestamp()))
        nonce = secrets.token_hex(16)
        assinatura = self._key.sign(
            canonical_request(method, caminho, query, timestamp, nonce, body)
        ).hex()
        headers = {
            "X-Valtexa-Key-Id": self.identity.key_id,
            "X-Valtexa-Timestamp": timestamp,
            "X-Valtexa-Nonce": nonce,
            "X-Valtexa-Signature": assinatura,
        }
        if content_type:
            headers["Content-Type"] = content_type
        headers.update(extra_headers or {})
        cabecalhos = sum(len(k) + len(v) + 4 for k, v in headers.items())
        try:
            resposta = self.network.send(method.upper(), path, body, headers)
        except NetworkUnavailableError as exc:
            # Caiu depois de abrir a conexão: o corpo pode ter saído pela Starlink. Conta
            # (o chunk que esgota o tempo e é reenviado gasta duas vezes — e assim aparece).
            if self.contador is not None and getattr(exc, "pode_ter_enviado", False):
                self.contador(method.upper(), path, len(body) + cabecalhos, 0)
            raise
        if self.contador is not None:
            self.contador(method.upper(), path, len(body) + cabecalhos, len(resposta.body or b""))
        return resposta
