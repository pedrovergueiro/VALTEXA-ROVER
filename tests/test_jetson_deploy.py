"""O pacote de instalação da Jetson (``deploy/jetson``) bate com o código."""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
JETSON = RAIZ / "deploy" / "jetson"


def _lista_bash(nome: str) -> set[str]:
    texto = (JETSON / "instalar.sh").read_text(encoding="utf-8")
    corpo = re.search(rf"^{nome}=\(([^)]*)\)", texto, re.M).group(1)
    return set(re.findall(r'"([^"]+)"', corpo))


@pytest.mark.skipif(sys.version_info < (3, 11), reason="tomllib é do 3.11")
def test_dependencias_do_instalador_batem_com_o_pyproject() -> None:
    import tomllib

    projeto = tomllib.loads((RAIZ / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    base = {d for d in projeto["dependencies"] if not d.startswith("onnxruntime")}
    assert _lista_bash("DEPENDENCIAS") == base  # o onnxruntime vem da NVIDIA, à parte
    assert _lista_bash("DEPENDENCIAS_ROVER_V3") == set(
        projeto["optional-dependencies"]["rover-v3"])


def test_scripts_em_lf_e_arquivos_referidos_existem() -> None:
    texto = (JETSON / "instalar.sh").read_bytes()
    assert b"\r\n" not in texto and texto.startswith(b"#!/usr/bin/env bash")
    for relativo in re.findall(r'"\$REPO/([^"]+)"', texto.decode()):
        assert (RAIZ / relativo).exists(), relativo


def test_hal_do_rover_env_existe_e_unidade_libera_o_hardware() -> None:
    env = dict(linha.split("=", 1) for linha in (JETSON / "rover.env").read_text(
        encoding="utf-8").splitlines() if linha and not linha.startswith("#"))
    modulo, _, funcao = env["VALTEXA_HAL"].partition(":")
    assert callable(getattr(importlib.import_module(modulo), funcao))
    dropin = (JETSON / "valtexa-rover.service.d" / "rover-v3.conf").read_text(encoding="utf-8")
    assert "SupplementaryGroups=video render dialout" in dropin
    assert "RuntimeDirectory=valtexa" in dropin  # /run/valtexa da navegação
    exemplo = (RAIZ / "deploy" / "hardware.rover-v3.json.exemplo").read_text(encoding="utf-8")
    assert "/run/valtexa/" in exemplo and "/dev/valtexa-gnss" in exemplo
    assert 'SYMLINK+="valtexa-gnss"' in (JETSON / "99-valtexa-gnss.rules").read_text()
