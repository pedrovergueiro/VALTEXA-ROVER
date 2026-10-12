"""As cinco conferências de um comando assinado (protocolo, seção 6)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from valtexa_rover._compat import UTC
from valtexa_rover.runtime.mission import CommandRejectedError, verify_command
from valtexa_rover.runtime.store import LocalStore

AGORA = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)


@pytest.fixture
def plataforma():
    chave = Ed25519PrivateKey.generate()
    return chave, chave.public_key().public_bytes_raw().hex()


def envelope(chave, **muda) -> dict:
    payload = {"device_id": "dev-1", "sequence": 1, "nonce": "n1",
               "expires_at": (AGORA + timedelta(hours=1)).isoformat(),
               "body": {"mission_id": "m-1", "field_id": "t-1"}, **muda}
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {"payload": payload, "signature": chave.sign(canon).hex()}


def test_comando_valido_e_repeticao_recusada(tmp_path, plataforma) -> None:
    chave, publica = plataforma
    store = LocalStore(tmp_path)
    env = envelope(chave)
    corpo = verify_command(env, device_id="dev-1", platform_public_key=publica, store=store,
                           now=AGORA)
    assert corpo["mission_id"] == "m-1"
    with pytest.raises(CommandRejectedError, match="sequência"):
        verify_command(env, device_id="dev-1", platform_public_key=publica, store=store,
                       now=AGORA)
    with pytest.raises(CommandRejectedError, match="nonce"):
        verify_command(envelope(chave, sequence=2), device_id="dev-1",
                       platform_public_key=publica, store=store, now=AGORA)


@pytest.mark.parametrize(
    ("muda", "adultera", "trecho"),
    [
        ({}, True, "assinatura"),
        ({"device_id": "dev-2"}, False, "outro robô"),
        ({"expires_at": (AGORA - timedelta(seconds=1)).isoformat()}, False, "vencido"),
    ],
)
def test_comando_recusado(tmp_path, plataforma, muda, adultera, trecho) -> None:
    chave, publica = plataforma
    env = envelope(chave, **muda)
    if adultera:
        env["payload"]["body"]["field_id"] = "outro"
    with pytest.raises(CommandRejectedError, match=trecho):
        verify_command(env, device_id="dev-1", platform_public_key=publica,
                       store=LocalStore(tmp_path), now=AGORA)
