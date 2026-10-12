"""Edge AI com cabeça sobre a torre de visão — o caminho do café.

No café a CNN sozinha acerta 6,5% num acervo externo; quem diagnostica é a cabeça
(``<modelo>.head.npz``) sobre o embedding da torre de visão do SigLIP 2
(``<modelo>.encoder.onnx``). Aqui a torre é minúscula (média RGB por canal) e a cabeça
lê a cor: o que se testa é o caminho — a conferência da torre pelo SHA-256, a mistura
sem a CNN, o recorte, a decisão e o staging com a torre ao lado.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from conftest import LABELS, build_tiny_onnx
from test_edge_perfil import PERFIL, quadro

from valtexa_rover import cli
from valtexa_rover.config import Category, ClassificationConfig
from valtexa_rover.edge_ai.engine import EdgeAIEngine, EdgeAIError, ModelPackage
from valtexa_rover.edge_ai.perfil import EdgeProfile, ProfileError
from valtexa_rover.edge_ai.vendor.heads import DiagnosisHead, EmbeddingMLP
from valtexa_rover.hal.local import OnnxCpuAccelerator

TORRE = "teste/torre-rgb"


def torre_minuscula(destino: Path) -> str:
    """NCHW (b, 3, 224, 224) → média de cada canal (b, 3). Devolve o SHA-256."""
    import onnx
    from onnx import TensorProto, helper

    grafo = helper.make_graph(
        [helper.make_node("ReduceMean", ["pixel_values"], ["emb"], axes=[2, 3], keepdims=0)],
        "torre_teste",
        [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT,
                                       ["batch", 3, 224, 224])],
        [helper.make_tensor_value_info("emb", TensorProto.FLOAT, ["batch", 3])],
    )
    modelo = helper.make_model(grafo, opset_imports=[helper.make_opsetid("", 17)])
    modelo.ir_version = 8
    onnx.checker.check_model(modelo)
    destino.write_bytes(modelo.SerializeToString())
    return hashlib.sha256(destino.read_bytes()).hexdigest()


def cabeca_de_cor(destino: Path, mix: dict[str, float] | None = None) -> None:
    """Lê a cor do embedding normalizado: verde = Healthy, vermelho = Rust, azul = minador."""
    k = 10.0
    pesos = np.array([[-k, k, -k], [k, -k, -k], [-k, -k, k], [0.0, 0.0, 0.0]])
    mlp = EmbeddingMLP(mean=np.zeros(3), std=np.ones(3), weights=(), biases=(),
                       activations=(), out_weight=pesos, out_bias=np.zeros(4))
    DiagnosisHead(class_names=tuple(LABELS), mlp=mlp, encoder=TORRE,
                  mix=mix or {"cnn": 0.0, "head": 1.0}).save(destino)


def modelo_com_cabeca(pasta: Path, *, sha_torre: str | None = None, mix=None) -> Path:
    onnx = pasta / "cafe.onnx"
    build_tiny_onnx(onnx)
    sha = torre_minuscula(onnx.with_suffix(".encoder.onnx"))
    cabeca_de_cor(onnx.with_suffix(".head.npz"), mix)
    perfil = {**PERFIL, "crop": "coffee",
              "head": {"encoder_model_id": TORRE, "encoder_sha256": sha_torre or sha,
                       "input_size": 224}}
    onnx.with_suffix(".edge.json").write_text(json.dumps(perfil), encoding="utf-8")
    return onnx


def motor(onnx: Path) -> EdgeAIEngine:
    m = EdgeAIEngine(OnnxCpuAccelerator(), ClassificationConfig())
    m.load(ModelPackage("coffee-foliar", "1", hashlib.sha256(onnx.read_bytes()).hexdigest(),
                        onnx))
    return m


def test_cabeca_decide_sem_rodar_a_cnn(tmp_path) -> None:
    m = motor(modelo_com_cabeca(tmp_path))
    assert m.head is not None

    def cnn_proibida(*_a, **_k):
        raise AssertionError("com peso zero na mistura, a CNN não roda")

    m._model.__class__ = type("Proibida", (m._model.__class__,), {"__call__": cnn_proibida})
    r = m.infer(quadro(True), distance_mm=270.0, camera_height_mm=1000.0)
    assert r.category is Category.POSSIVEL_DOENCA and r.label == "Rust"
    zonas = {z for t in r.tiles for z in t.zones}  # os terços do cafeeiro
    assert "terco_medio" in zonas and zonas <= {"terco_inferior", "terco_medio", "terco_superior"}
    sadio = m.infer(quadro(False), distance_mm=270.0, camera_height_mm=1000.0)
    assert sadio.category is Category.NORMAL and not sadio.low_confidence


def test_mistura_com_a_cnn_usa_as_duas(tmp_path) -> None:
    m = motor(modelo_com_cabeca(tmp_path, mix={"cnn": 0.5, "head": 0.5}))
    r = m.infer(quadro(True), distance_mm=270.0)
    assert r.label == "Rust"
    probs = np.array([t.probabilities for t in r.tiles])
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)


def test_torre_errada_ou_faltando_e_recusada(tmp_path) -> None:
    with pytest.raises(EdgeAIError, match="SHA-256"):
        motor(modelo_com_cabeca(tmp_path / "a", sha_torre="0" * 64))
    onnx = modelo_com_cabeca(tmp_path / "b")
    onnx.with_suffix(".head.npz").unlink()
    with pytest.raises(EdgeAIError, match=r"head\.npz"):
        motor(onnx)
    onnx = modelo_com_cabeca(tmp_path / "c")
    outra = {**json.loads(onnx.with_suffix(".edge.json").read_text())}
    outra["head"] = {**outra["head"], "encoder_model_id": "google/outra-torre"}
    onnx.with_suffix(".edge.json").write_text(json.dumps(outra))
    with pytest.raises(EdgeAIError, match="treinada sobre"):
        motor(onnx)
    with pytest.raises(ProfileError):
        EdgeProfile.from_dict({**PERFIL, "head": {"encoder_model_id": "x",
                                                 "encoder_sha256": "nao-e-hash"}})


def test_preparar_modelo_leva_a_torre_e_a_cabeca(tmp_path, capsys) -> None:
    onnx = modelo_com_cabeca(tmp_path / "origem")
    torre = onnx.with_suffix(".encoder.onnx")
    fora = tmp_path / "framing_vision.onnx"
    torre.replace(fora)  # como vem do ia/: com outro nome, longe do modelo
    dados = tmp_path / "dados"
    assert cli.main(["preparar-modelo", str(onnx), "--dados", str(dados), "--quadros", "0"]) == 1
    assert "encoder.onnx" in capsys.readouterr().err
    assert not [p for p in (dados / "ota" / "models").iterdir() if not p.name.startswith(".")]

    assert cli.main(["preparar-modelo", str(onnx), "--dados", str(dados), "--quadros", "1",
                     "--encoder", str(fora)]) == 0
    saida = json.loads(capsys.readouterr().out)
    sha = saida["sha256"]
    assert saida["head"] is True and saida["medicao"]["recortes"] > 20
    assert {p.name for p in (dados / "ota" / "models").iterdir()} == {
        f"{sha}.onnx", f"{sha}.labels.json", f"{sha}.edge.json", f"{sha}.head.npz",
        f"{sha}.encoder.onnx"}
