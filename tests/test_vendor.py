"""A cópia do pré-processamento e do runtime ONNX precisa ser idêntica à do ``ia/``."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from valtexa_rover.edge_ai import vendor

RAIZ = Path(__file__).resolve().parents[2]
AQUI = Path(vendor.__file__).parent


def _hash(caminho: Path) -> str:
    """CRLF/LF não conta: no Windows o checkout pode trazer CRLF, e o git normaliza no commit."""
    return hashlib.sha256(caminho.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _origem(relativo: str) -> Path:
    """``ia/...`` no repositório da nuvem; ``../VALTEXA-B/...`` com os repositórios separados."""
    monorepo = RAIZ / relativo
    if monorepo.exists():
        return monorepo
    return RAIZ / "VALTEXA-B" / relativo.removeprefix("ia/")


@pytest.mark.parametrize("nome", sorted(vendor.ORIGENS))
def test_copia_identica_ao_ia(nome: str) -> None:
    origem = _origem(vendor.ORIGENS[nome])
    if not origem.exists():
        pytest.skip("ia/ não está ao lado (pacote instalado sozinho)")
    assert _hash(AQUI / nome) == _hash(origem), (
        f"{nome} divergiu de {vendor.ORIGENS[nome]}: copie de novo, sem editar"
    )


def test_pacote_nao_importa_saas_nem_ia() -> None:
    pacote = AQUI.parents[1]
    for arquivo in pacote.rglob("*.py"):
        texto = arquivo.read_text(encoding="utf-8")
        for proibido in ("from valtexa.", "import valtexa\n", "from app.", "import app."):
            assert proibido not in texto, f"{arquivo} importa {proibido!r}"
