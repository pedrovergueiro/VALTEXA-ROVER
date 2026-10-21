"""Nível 2, o que faltava: dúvida pelo conformal e pela régua da captura, vista nova conferida,
região reobservável, desacelerar só quando a navegação deixa, luz auxiliar — e a geometria 3D
de cada recorte (ponto, orientação da folha, face foliar indeterminada)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from test_percepcao import AMBIGUA, cena, motor, primeira, relogio  # noqa: F401

from valtexa_rover.config import ActivePerceptionConfig
from valtexa_rover.edge_ai.geometria import FACE_INDETERMINADA, geometria_do_recorte
from valtexa_rover.hal.simulated import SimulatedController, SimulatedLight
from valtexa_rover.runtime.percepcao import (
    ActivePerception,
    Uncertainty,
    assess,
    deslocamento_na_foto,
    pode_reduzir,
    reobservavel,
)

LABELS = ["Healthy", "Rust", "Leaf_Miner_Damage", "Unknown_Background"]
LIBERADO = {"modo": "autonomo", "reducao_permitida": True, "speed_ms": 0.1}


def _qualidade(caminho, **limites) -> None:
    ref = {"img_size": 32, "min_sharpness": 0.0, "min_brightness": 0.0, "max_brightness": 256.0,
           "min_contrast": 0.0, "max_highlights": 1.0, "percentile": 0.5, "fitted_on": 1000,
           **limites}
    caminho.with_suffix(".quality.json").write_text(json.dumps(ref), encoding="utf-8")


def _conformal(caminho, limiar: float, classes=LABELS) -> None:
    ref = {"class_names": classes, "alpha": 0.1,
           "thresholds": {"plain": [limiar] * len(classes), "tta": [limiar] * len(classes)},
           "calibrated_on": 500, "class_counts": [100] * len(classes)}
    caminho.with_suffix(".conformal.json").write_text(json.dumps(ref), encoding="utf-8")


def _motor(tmp_path, **refs):
    m = motor(tmp_path)  # grava m.onnx e carrega
    caminho = tmp_path / "m.onnx"
    if "qualidade" in refs:
        _qualidade(caminho, **refs["qualidade"])
    if "conformal" in refs:
        _conformal(caminho, refs["conformal"], refs.get("classes", LABELS))
    m.load(m.package)  # recarrega com as referências
    return m


# ---------------------------------------------------------------------------------------------
# Geometria por recorte
# ---------------------------------------------------------------------------------------------


def _plano(angulo_graus: float, z0: float = 300.0, n: int = 120) -> np.ndarray:
    """Mapa de profundidade de um plano inclinado em torno do eixo vertical da foto."""
    f = n / (2 * math.tan(math.radians(90.0) / 2))
    u = np.arange(n) + 0.5 - n / 2
    t = math.tan(math.radians(angulo_graus))
    # z = z0 + t·x, com x = u·z/f  =>  z = z0 / (1 - t·u/f)
    linha = z0 / (1 - t * u / f)
    return np.tile(linha, (n, 1))


def test_inclinacao_da_folha_pelo_plano() -> None:
    for angulo in (0.0, 30.0, 60.0):
        g = geometria_do_recorte(_plano(angulo), (120, 120), (40, 40, 80, 80), 90.0)
        assert g is not None and g["superficie"] == "plana"
        assert g["inclinacao_graus"] == pytest.approx(angulo, abs=1.0)
        assert g["face"] == FACE_INDETERMINADA  # a profundidade não diz a face
        assert g["normal"][2] < 0  # virada para a câmera
    frente = geometria_do_recorte(_plano(0.0), (120, 120), (55, 55, 65, 65), 90.0)
    assert frente["centro_mm"][0] == pytest.approx(0.0, abs=1.0)
    assert frente["centro_mm"][2] == pytest.approx(300.0, abs=0.5)


def test_superficie_irregular_e_falta_de_medida() -> None:
    rng = np.random.default_rng(0)
    caos = 300.0 + rng.uniform(-80, 80, (120, 120))
    g = geometria_do_recorte(caos, (120, 120), (20, 20, 100, 100), 90.0)
    assert g["superficie"] == "irregular" and g["normal"] is None and g["inclinacao_graus"] is None
    vazio = np.zeros((120, 120))
    assert geometria_do_recorte(vazio, (120, 120), (20, 20, 100, 100), 90.0) is None
    # A faixa da planta tira o fundo do plano.
    fundo = _plano(0.0)
    fundo[:, :60] = 2500.0
    so_planta = geometria_do_recorte(fundo, (120, 120), (20, 20, 100, 100), 90.0,
                                     faixa_mm=(200.0, 500.0))
    assert so_planta["superficie"] == "plana" and so_planta["pixels_medidos"] == pytest.approx(
        0.5, abs=0.05)


def test_recortes_saem_com_geometria_com_mapa(tmp_path, relogio) -> None:  # noqa: F811
    m = motor(tmp_path)
    _quadro, r = primeira(m, cena(relogio, cor=AMBIGUA))
    assert r.tiles and all(t.geometria is not None for t in r.tiles if t.depth_mm)
    assert {t.geometria["face"] for t in r.tiles if t.geometria} == {FACE_INDETERMINADA}


# ---------------------------------------------------------------------------------------------
# Régua da captura e conformal no motor
# ---------------------------------------------------------------------------------------------


def test_motor_mede_qualidade_so_dos_que_importam_e_conformal_de_todos(tmp_path,
                                                                     relogio) -> None:  # noqa: F811
    m = _motor(tmp_path, qualidade={"min_sharpness": 1e9}, conformal=1.0)
    _quadro, r = primeira(m, cena(relogio, cor=AMBIGUA))
    medidos = [t for t in r.tiles if t.quality is not None]
    assert 0 < len(medidos) <= 8 and all("blurry" in t.quality for t in medidos)
    assert all(t.conformal_set is not None and len(t.conformal_set) == len(LABELS)
               for t in r.tiles)


def test_referencia_que_nao_bate_e_ignorada_e_avisada(tmp_path) -> None:
    m = _motor(tmp_path, qualidade={"img_size": 224}, conformal=0.5,
               classes=["a", "b", "c", "d"])
    assert m.quality is None and m.conformal is None and len(m.avisos) == 2


def test_duvida_pelo_conformal_e_pela_qualidade(tmp_path, relogio) -> None:  # noqa: F811
    cfg = ActivePerceptionConfig()
    amplo = _motor(tmp_path / "a", conformal=1.0)  # todas as classes no conjunto
    quadro, r = primeira(amplo, cena(relogio, cor=AMBIGUA))
    d = assess(r, quadro, amplo, cfg)
    assert "conformal_ambiguo" in d.reasons and d.decision_doubt
    estreito = _motor(tmp_path / "b", conformal=0.0)  # nenhuma classe: nunca ambíguo
    quadro, r = primeira(estreito, cena(relogio, cor=AMBIGUA))
    assert "conformal_ambiguo" not in assess(r, quadro, estreito, cfg).reasons
    borrado = _motor(tmp_path / "c", qualidade={"min_sharpness": 1e9})
    quadro, r = primeira(borrado, cena(relogio, cor=AMBIGUA))
    d = assess(r, quadro, borrado, cfg)
    assert "qualidade_baixa" in d.reasons and d.problemas[0] == "blurry"
    # Qualidade é do sensor: sozinha não manda original para o agrônomo.
    assert not Uncertainty(("qualidade_baixa",), None).decision_doubt
    assert Uncertainty(("conformal_ambiguo",), None).decision_doubt


# ---------------------------------------------------------------------------------------------
# Reobservável, desacelerar, luz e a vista nova conferida
# ---------------------------------------------------------------------------------------------


def test_deslocamento_e_reobservavel() -> None:
    # 1 m/s, 0,4 s, planta a 27 cm com 127° de campo: a foto cobre ~1,08 m; anda ~37%.
    d = deslocamento_na_foto(1.0, 0.4, 270.0, 127.0)
    assert d == pytest.approx(0.4 / (2 * 0.27 * math.tan(math.radians(63.5))), rel=1e-6)
    assert reobservavel((0.4, 0.3, 0.6, 0.7), 0.1) and not reobservavel((0.4, 0.3, 0.6, 0.7), 0.5)
    assert reobservavel(None, 0.3) and not reobservavel(None, 0.6)


def test_so_desacelera_quando_a_navegacao_deixa() -> None:
    assert pode_reduzir(LIBERADO)
    for proibido in ({"modo": "teleop"}, {"teleop": True}, {"obstaculo": True},
                     {"travessia": True}, {"reducao_permitida": False}):
        assert not pode_reduzir({**LIBERADO, **proibido})
    assert not pode_reduzir({})  # sem dado, não pode


def test_rapido_demais_desacelera_e_ainda_olha(tmp_path, relogio) -> None:  # noqa: F811
    m = motor(tmp_path)
    camera = cena(relogio, cor=AMBIGUA)
    cfg = ActivePerceptionConfig(reduzir_velocidade=True)
    nav = SimulatedController({**LIBERADO, "speed_ms": 1.2})
    d = ActivePerception(cfg, relogio).run(m, camera, *primeira(m, camera), controller=nav)
    assert nav.velocidades and nav.velocidades[0]["seconds"] == 1.5
    assert d.block["velocidade"]["fator"] == 0.3 and d.block["views"]
    # Sem a navegação liberar, não pede — e rápido demais, não insiste.
    preso = SimulatedController({"modo": "teleop", "speed_ms": 1.2})
    d = ActivePerception(cfg, relogio).run(m, camera, *primeira(m, camera), controller=preso)
    assert not preso.velocidades and d.block["skipped"] == "fora_do_campo"
    assert d.block["views"] == []


def test_luz_sobe_no_escuro_e_volta(tmp_path, relogio) -> None:  # noqa: F811
    m = _motor(tmp_path, qualidade={"min_brightness": 250.0})
    camera = cena(relogio, cor=AMBIGUA)
    luz = SimulatedLight(0.2)
    d = ActivePerception(ActivePerceptionConfig(), relogio).run(m, camera, *primeira(m, camera),
                                                                luz=luz)
    assert d.block["luz"] == pytest.approx(0.5) and luz.level() == pytest.approx(0.2)
    assert luz.historico == pytest.approx([0.5, 0.2])


def test_vista_nova_fora_da_regua_nao_entra_na_fusao(tmp_path, relogio) -> None:  # noqa: F811
    m = _motor(tmp_path, qualidade={"min_sharpness": 1e9})
    camera = cena(relogio, cor=AMBIGUA)
    ap = ActivePerception(ActivePerceptionConfig(max_vistas_extras=2), relogio)
    d = ap.run(m, camera, *primeira(m, camera))
    assert d.block["views"] and all(v.get("quality") == "ruim" for v in d.block["views"])
    assert "fused_tiles" not in d.block and ap.snapshot()["vista_ruim"] == 2


# ---------------------------------------------------------------------------------------------
# 5.1: cabeça adaptada no campo só roda com procedência
# ---------------------------------------------------------------------------------------------


def _adaptada(tmp_path, registro: dict | None, carimbo: str | None = "certo"):
    """O modelo com cabeça, o perfil dizendo que ela veio do contínuo e a procedência ao lado."""
    import hashlib

    from test_edge_cabeca import TORRE, modelo_com_cabeca

    onnx = modelo_com_cabeca(tmp_path)
    sha = hashlib.sha256(onnx.with_suffix(".head.npz").read_bytes()).hexdigest()
    proc = onnx.with_suffix(".head.provenance.json")
    if registro is not None:
        base = {"formato": 1, "tipo": "cabeca", "modelo": "coffee-foliar", "encoder": TORRE,
                "cabeca": sha, "cabeca_base": "a" * 64, "rodada": "r1", "escopo": "T01",
                "quando": "2026-10-18T00:00:00Z", "aprovado_por": "Eng. Agr. Teste",
                "portao": {"decisao": "APROVADA", "motivos": [], "laudo_sha256": "c" * 64},
                "dados": {"campo_sha256": "d" * 64}, "motor": {}, "forcada": False,
                "motivo": None}
        proc.write_text(json.dumps({**base, **registro}), encoding="utf-8")
    perfil = json.loads(onnx.with_suffix(".edge.json").read_text(encoding="utf-8"))
    perfil["head"]["head_sha256"] = sha
    carimbado = (hashlib.sha256(proc.read_bytes()).hexdigest() if carimbo == "certo"
                 and proc.exists() else carimbo)
    perfil["measured"] = {"continuo": {"rodada": "r1", "cabeca": sha,
                                       "procedencia_sha256": carimbado}}
    onnx.with_suffix(".edge.json").write_text(json.dumps(perfil), encoding="utf-8")
    return onnx


def _carrega(onnx):
    import hashlib

    from valtexa_rover.config import ClassificationConfig
    from valtexa_rover.edge_ai.engine import EdgeAIEngine, ModelPackage
    from valtexa_rover.hal.local import OnnxCpuAccelerator

    m = EdgeAIEngine(OnnxCpuAccelerator(), ClassificationConfig())
    m.load(ModelPackage("coffee-foliar", "1", hashlib.sha256(onnx.read_bytes()).hexdigest(),
                        onnx))
    return m


def test_cabeca_adaptada_com_procedencia_roda(tmp_path) -> None:
    m = _carrega(_adaptada(tmp_path, {}))
    assert m.head is not None and m.procedencia["aprovado_por"] == "Eng. Agr. Teste"


@pytest.mark.parametrize(("registro", "carimbo", "trecho"), [
    (None, None, "falta"),  # o perfil diz contínuo e não veio procedência
    ({}, "0" * 64, "carimbou"),  # trocaram a procedência
    ({"cabeca": "e" * 64}, "certo", "outra cabeça"),
    ({"aprovado_por": ""}, "certo", "aprovador"),
    ({"portao": {"decisao": "REPROVADA", "motivos": [], "laudo_sha256": "c" * 64}}, "certo",
     "REPROVADA"),
    ({"encoder": "outra/torre"}, "certo", "torre"),
])
def test_cabeca_adaptada_sem_procedencia_valida_nao_roda(tmp_path, registro, carimbo,
                                                         trecho) -> None:
    from valtexa_rover.edge_ai.engine import EdgeAIError

    with pytest.raises(EdgeAIError, match=trecho):
        _carrega(_adaptada(tmp_path, registro, carimbo))


def test_cabeca_de_origem_nao_precisa_de_procedencia(tmp_path) -> None:
    from test_edge_cabeca import modelo_com_cabeca

    m = _carrega(modelo_com_cabeca(tmp_path))
    assert m.head is not None and m.procedencia is None
