"""DUBLÊ da nuvem em memória, com as regras do SaaS para o que o robô envia.

Idempotência por ID e hash, chunks com retomada, telemetria, sessão e
designação de modelos — o suficiente para exercer o Runtime sem a plataforma
(testes do pacote, simulador e estresse). **Não é a plataforma**: o teste contra
o SaaS de verdade está em ``saas/tests/test_rover_integracao.py`` e
``saas/tests/test_ponta_a_ponta.py``.
"""

from __future__ import annotations

import hashlib
import json
import math

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from valtexa_rover.hal.interfaces import HttpResponse
from valtexa_rover.runtime.client import canonical_request


class NuvemEmMemoria:
    """DUBLÊ da nuvem (o teste real é contra o SaaS)."""

    def __init__(self, public_key_hex: str, designations: list[dict]) -> None:
        self.public_key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
        self.designations = designations
        self.records: dict[str, str] = {}
        self.record_bodies: dict[str, dict] = {}
        self.record_posts = 0
        self.duplicated = 0
        self.uploads: dict[str, dict] = {}
        self.objects: dict[str, bytes] = {}
        self.telemetry: dict[int, dict] = {}
        self.sessions: list[dict] = []
        self.chunk_puts = 0
        self.fail_queue: list[int] = []
        self.remaining_seen: list[int] = []

    def __call__(self, method: str, path: str, body: bytes, headers: dict) -> HttpResponse:
        mensagem = canonical_request(method, path.partition("?")[0], "",
                                     headers["X-Valtexa-Timestamp"],
                                     headers["X-Valtexa-Nonce"], body)
        self.public_key.verify(bytes.fromhex(headers["X-Valtexa-Signature"]), mensagem)
        if headers.get("Content-Encoding") == "gzip":  # assinado comprimido, como na plataforma
            import gzip

            self.bytes_comprimidos = getattr(self, "bytes_comprimidos", 0) + len(body)
            body = gzip.decompress(body)
        if self.fail_queue:
            return HttpResponse(self.fail_queue.pop(0), b'{"detail": "falha injetada"}')
        partes = path.strip("/").split("/")
        if path == "/v1/device/models":
            return _json(200, self.designations)
        if path == "/v1/device/session":
            self.sessions.append(json.loads(body))
            return _json(200, {"access_token": "x"})
        if path == "/v1/device/sync":
            dados = json.loads(body)
            for amostra in dados["samples"]:
                self.telemetry[amostra["sequence_number"]] = amostra
            self.remaining_seen.append(dados.get("remaining", 0))
            return _json(200, {"accepted": len(dados["samples"])})
        if path == "/v1/device/edge/records":
            return self._records(json.loads(body))
        if path == "/v1/device/uploads":
            return self._open(json.loads(body))
        if partes[:3] == ["v1", "device", "uploads"] and len(partes) == 6:
            up = self.uploads[partes[3]]
            up["chunks"][int(partes[5])] = body
            self.chunk_puts += 1
            return _json(200, {"received": len(up["chunks"])})
        if partes[:3] == ["v1", "device", "uploads"] and partes[-1] == "complete":
            up = self.uploads[partes[3]]
            dados = b"".join(up["chunks"][i] for i in range(up["total"]))
            if hashlib.sha256(dados).hexdigest() != up["sha256"]:
                up["chunks"].clear()
                return _json(422, {"detail": "hash não confere"})
            self.objects[up["sha256"]] = dados
            up["status"] = "complete"
            return _json(200, {"status": "complete"})
        return _json(404, {"detail": path})

    def _records(self, dados: dict) -> HttpResponse:
        self.record_posts += 1
        self.remaining_seen.append(dados.get("remaining", 0))
        saida = {"accepted": [], "duplicated": [], "conflicts": [], "rejected": []}
        for r in dados["records"]:
            calculado = hashlib.sha256(
                json.dumps(r["body"], sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False).encode()
            ).hexdigest()
            if calculado != r["sha256"]:
                saida["rejected"].append({"id": r["id"], "error": "hash não confere"})
            elif r["id"] in self.records:
                chave = "duplicated" if self.records[r["id"]] == r["sha256"] else "conflicts"
                self.duplicated += chave == "duplicated"
                saida[chave].append(r["id"])
            else:
                self.records[r["id"]] = r["sha256"]
                self.record_bodies[r["id"]] = r["body"]
                saida["accepted"].append(r["id"])
        return _json(200, saida)

    def _open(self, dados: dict) -> HttpResponse:
        if dados["sha256"] in self.objects:
            return _json(200, {"upload_id": dados["upload_id"], "status": "complete",
                               "received_chunks": [], "chunk_size": dados["chunk_size"],
                               "total_chunks": 1})
        up = self.uploads.setdefault(dados["upload_id"], {
            "sha256": dados["sha256"], "chunk_size": dados["chunk_size"],
            "total": max(1, math.ceil(dados["size_bytes"] / dados["chunk_size"])),
            "chunks": {}, "status": "receiving", "meta": dados,
        })
        return _json(200, {"upload_id": dados["upload_id"], "status": up["status"],
                           "received_chunks": sorted(up["chunks"]),
                           "chunk_size": up["chunk_size"], "total_chunks": up["total"]})


def _json(status: int, data) -> HttpResponse:
    return HttpResponse(status, json.dumps(data).encode())


__all__ = ["NuvemEmMemoria"]
