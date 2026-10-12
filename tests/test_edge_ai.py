from __future__ import annotations

import json

import pytest
from conftest import build_tiny_onnx, image_bytes

from valtexa_rover.config import Category, ClassificationConfig
from valtexa_rover.edge_ai.engine import EdgeAIEngine, EdgeAIError, ModelPackage, categoria_padrao
from valtexa_rover.edge_ai.vendor.preprocess import InvalidImageError
from valtexa_rover.hal.local import OnnxCpuAccelerator


@pytest.fixture
def engine(tmp_path):
    caminho = tmp_path / "m.onnx"
    sha = build_tiny_onnx(caminho)
    motor = EdgeAIEngine(OnnxCpuAccelerator(), ClassificationConfig())
    motor.load(ModelPackage("coffee-foliar", "2.4.1", sha, caminho))
    return motor


@pytest.mark.parametrize(
    ("cor", "rotulo", "categoria", "baixa"),
    [
        ("verde", "Healthy", Category.NORMAL, False),
        ("vermelho", "Rust", Category.POSSIVEL_DOENCA, False),
        ("azul", "Leaf_Miner_Damage", Category.POSSIVEL_PRAGA, False),
        ("cinza", None, Category.ANOMALIA, True),
        ("duvida", "Healthy", Category.NORMAL, True),
    ],
)
def test_classificacao_pelo_onnx_runtime(engine, cor, rotulo, categoria, baixa) -> None:
    r = engine.infer(image_bytes(cor))
    assert r.category is categoria
    assert r.low_confidence is baixa
    if rotulo:
        assert r.label == rotulo
    assert r.input_size == 32 and r.latency_ms > 0
    assert len(r.top) == 3 and r.top[0][1] >= r.top[1][1]
    if categoria is Category.ANOMALIA:
        assert "nenhuma classe reconhecida" in r.reason


def test_estatisticas_para_a_saude(engine) -> None:
    for _ in range(5):
        engine.infer(image_bytes("verde"))
    assert engine.stats.inferences == 5
    assert engine.stats.fps and engine.stats.fps > 0


def test_imagem_ilegivel_conta_erro(engine) -> None:
    with pytest.raises(InvalidImageError):
        engine.infer(b"nao e imagem")
    assert engine.stats.errors == 1


def test_categorias_do_pacote_do_modelo_valem_sobre_a_deducao(tmp_path) -> None:
    caminho = tmp_path / "m.onnx"
    sha = build_tiny_onnx(caminho)
    caminho.with_suffix(".categories.json").write_text(json.dumps({"Rust": "ANOMALIA"}))
    motor = EdgeAIEngine(OnnxCpuAccelerator(), ClassificationConfig())
    motor.load(ModelPackage("k", "1", sha, caminho))
    assert motor.infer(image_bytes("vermelho")).category is Category.ANOMALIA


def test_rotulos_que_nao_batem_com_o_modelo_sao_recusados(tmp_path) -> None:
    caminho = tmp_path / "m.onnx"
    sha = build_tiny_onnx(caminho)
    caminho.with_suffix(".labels.json").write_text(json.dumps(["a", "b"]))
    motor = EdgeAIEngine(OnnxCpuAccelerator(), ClassificationConfig())
    with pytest.raises(EdgeAIError, match="4 saídas"):
        motor.load(ModelPackage("k", "1", sha, caminho))
    assert not motor.loaded


def test_arquivo_que_nao_e_onnx_e_recusado(tmp_path) -> None:
    caminho = tmp_path / "m.onnx"
    caminho.write_bytes(b"lixo")
    caminho.with_suffix(".labels.json").write_text("[]")
    motor = EdgeAIEngine(OnnxCpuAccelerator(), ClassificationConfig())
    with pytest.raises(EdgeAIError, match="ONNX Runtime recusou"):
        motor.load(ModelPackage("k", "1", "x", caminho))


@pytest.mark.parametrize(
    ("rotulo", "categoria"),
    [("Healthy", Category.NORMAL), ("Leaf_Miner_Damage", Category.POSSIVEL_PRAGA),
     ("Cercospora_Leaf_Spot", Category.POSSIVEL_DOENCA), ("Suspect", Category.ANOMALIA),
     ("Healthy_Canopy", Category.NORMAL), ("Sigatoka", Category.POSSIVEL_DOENCA)],
)
def test_deducao_de_categoria_pelos_rotulos_do_ia(rotulo, categoria) -> None:
    assert categoria_padrao(rotulo) is categoria
