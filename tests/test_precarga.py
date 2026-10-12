"""Pré-carga do modelo no boot e a regra da falha ``model_missing``."""

from __future__ import annotations

import json

from conftest import designation
from test_edge_perfil import PERFIL


def _reinicia_com(rig, designacoes: list) -> None:
    """O mesmo robô religado, com outra designação na plataforma."""
    rig.cloud.designations = designacoes
    rig.core.close()
    rig.core = rig.new_core()


def _faltas(rig) -> dict:
    rig.core.health.snapshot()
    return rig.core.health.faults


def test_boot_ja_deixa_o_modelo_designado_carregado(rig_factory) -> None:
    rig = rig_factory(["verde"])
    assert rig.core.engine.loaded and rig.core.engine.package.sha256 == rig.sha
    assert "model_missing" not in _faltas(rig)
    rig.core.start_mission("m-1", "talhao-1")  # começa sem carregar de novo
    assert rig.core.store.get("last_field_id") == "talhao-1"


def test_parado_sem_designacao_nao_e_erro(rig_factory) -> None:
    rig = rig_factory(["verde"])
    _reinicia_com(rig, [])
    assert not rig.core.engine.loaded
    assert "model_missing" not in _faltas(rig)  # disponível, esperando designação


def test_designado_que_nao_carrega_e_falha(rig_factory) -> None:
    rig = rig_factory(["verde"])
    _reinicia_com(rig, [designation("talhao-1", "f" * 64)])
    assert not rig.core.engine.loaded
    faltas = _faltas(rig)
    assert "model_missing" in faltas and "não está no robô" in faltas["model_missing"]


def test_perfil_recalibrado_com_o_mesmo_onnx_recarrega(rig_factory) -> None:
    rig = rig_factory(["verde"])
    antes = rig.core.engine.profile
    perfil = rig.core.models.models_dir / f"{rig.sha}.edge.json"
    perfil.write_text(json.dumps(PERFIL), encoding="utf-8")  # chegou pelo OTA
    rig.core.models.activate_for_field("talhao-1")
    assert rig.core.engine.profile is not None and rig.core.engine.profile != antes
