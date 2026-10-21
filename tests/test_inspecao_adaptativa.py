"""Inspeção adaptativa orientada por incerteza: o Rover reconhece quando não sabe.

A novidade no embedding da torre (``.novidade.npz``) marca o recorte longe de tudo o que a
cabeça conhece — e isso vale como dúvida mesmo com confiança alta. Cada quadro cai num dos
quatro casos (suficiente, anomalia confiante, ambíguo, informação insuficiente), e a térmica,
quando existe, entra como evidência sem decidir.
"""

from __future__ import annotations

import dataclasses
import io
import json
from datetime import datetime

import numpy as np
import pytest
from conftest import LABELS
from PIL import Image
from test_edge_cabeca import TORRE, modelo_com_cabeca
from test_percepcao import AMBIGUA, cena, motor, primeira

from valtexa_rover._compat import UTC
from valtexa_rover.clock import ManualClock
from valtexa_rover.config import ActivePerceptionConfig, Category, ClassificationConfig
from valtexa_rover.edge_ai.engine import EdgeAIEngine, ModelPackage
from valtexa_rover.edge_ai.vendor.novidade_cabeca import ReferenciaDeNovidade, ajusta
from valtexa_rover.hal.local import OnnxCpuAccelerator
from valtexa_rover.hal.simulated import SimulatedThermal
from valtexa_rover.runtime.percepcao import (
    ActivePerception,
    Uncertainty,
    assess,
    caso_de_inspecao,
    medida_termica,
)


@pytest.fixture
def relogio() -> ManualClock:
    return ManualClock(datetime(2026, 10, 19, 12, 0, tzinfo=UTC))


def _foto(cor, tamanho=(1600, 1200)) -> bytes:
    rgb = np.zeros((tamanho[1], tamanho[0], 3), np.uint8)
    rgb[...] = cor
    saida = io.BytesIO()
    Image.fromarray(rgb).save(saida, "JPEG", quality=95)
    return saida.getvalue()


def _carrega(onnx) -> EdgeAIEngine:
    import hashlib

    m = EdgeAIEngine(OnnxCpuAccelerator(), ClassificationConfig())
    m.load(ModelPackage("coffee-foliar", "1", hashlib.sha256(onnx.read_bytes()).hexdigest(),
                        onnx))
    return m


def _ancoras(m: EdgeAIEngine) -> dict[str, np.ndarray]:
    """O embedding (médio) que a torre de teste dá a cada cor pura — o centro de cada classe."""
    saida = {}
    for nome, cor in (("Healthy", (0, 255, 0)), ("Rust", (255, 0, 0)),
                      ("Leaf_Miner_Damage", (0, 0, 255)), ("Unknown_Background", (90, 90, 90))):
        m.infer(_foto(cor), distance_mm=270.0)
        saida[nome] = np.asarray(m._ultimos_embeddings, dtype=np.float64).mean(axis=0)
    return saida


def _referencia(ancoras: dict[str, np.ndarray], encoder: str = TORRE, ruido: float = 0.01,
                classes=LABELS) -> ReferenciaDeNovidade:
    rng = np.random.default_rng(0)
    x, y = [], []
    for k, nome in enumerate(classes):
        base = ancoras[nome] / np.linalg.norm(ancoras[nome])
        x.append(base + ruido * rng.normal(size=(60, len(base))))
        y += [k] * 60
    x, y = np.vstack(x), np.array(y)
    metade = np.arange(len(y)) % 2 == 0
    return ajusta(x[metade], y[metade], x[~metade], y[~metade], classes, encoder,
                  componentes=3)


# ---------------------------------------------------------------------------------------------
# A referência
# ---------------------------------------------------------------------------------------------


def test_referencia_marca_o_que_esta_longe_e_volta_do_arquivo(tmp_path) -> None:
    rng = np.random.default_rng(1)
    centros = rng.normal(size=(3, 16)) * 3
    x = np.vstack([c + rng.normal(size=(200, 16)) for c in centros])
    y = np.repeat(np.arange(3), 200)
    lado = rng.random(len(y))
    ref = ajusta(x[lado < 0.5], y[lado < 0.5], x[(lado >= 0.5) & (lado < 0.75)],
                 y[(lado >= 0.5) & (lado < 0.75)], ("a", "b", "c"), "t", componentes=8)
    teste = x[lado >= 0.75]
    assert ref.novos(teste).mean() < 0.05  # do domínio: perto do percentil prometido
    longe = rng.normal(size=(50, 16)) * 3 + 40.0
    assert ref.novos(longe).all()
    ref.save(tmp_path / "r.npz")
    de_novo = ReferenciaDeNovidade.load(tmp_path / "r.npz")
    assert np.allclose(de_novo.escore(teste), ref.escore(teste), rtol=1e-4)
    assert de_novo.medido["componentes"] == 8
    np.savez(tmp_path / "torta.npz", media=np.zeros(3))
    with pytest.raises(ValueError, match="ilegível"):
        ReferenciaDeNovidade.load(tmp_path / "torta.npz")


def test_classe_sem_exemplo_bastante_recusa() -> None:
    x = np.random.default_rng(0).normal(size=(30, 4))
    with pytest.raises(ValueError, match="sem exemplos"):
        ajusta(x, np.zeros(30, int), x, np.zeros(30, int), ("a", "b"), "t")


# ---------------------------------------------------------------------------------------------
# No motor
# ---------------------------------------------------------------------------------------------


def test_motor_pontua_cada_recorte(tmp_path) -> None:
    onnx = modelo_com_cabeca(tmp_path)
    m = _carrega(onnx)
    _referencia(_ancoras(m)).save(onnx.with_suffix(".novidade.npz"))
    m = _carrega(onnx)
    assert m.novidade is not None and not m.avisos
    conhecida = m.infer(_foto((255, 0, 0)), distance_mm=270.0)
    assert conhecida.label == "Rust" and conhecida.confidence > 0.9
    assert all(t.novidade is not None and t.novidade <= 1.0 for t in conhecida.tiles)
    # Amarelo: a cabeça ainda escolhe uma classe — e a novidade diz que ela não conhece isso.
    estranha = m.infer(_foto(AMBIGUA), distance_mm=270.0)
    assert all(t.novidade > 1.0 for t in estranha.tiles)


def test_referencia_de_outra_torre_e_ignorada(tmp_path) -> None:
    onnx = modelo_com_cabeca(tmp_path)
    m = _carrega(onnx)
    _referencia(_ancoras(m), encoder="outra/torre").save(onnx.with_suffix(".novidade.npz"))
    m = _carrega(onnx)
    assert m.novidade is None and any("novidade ignorada" in a for a in m.avisos)
    assert all(t.novidade is None for t in m.infer(_foto((255, 0, 0)), 270.0).tiles)


# ---------------------------------------------------------------------------------------------
# Na percepção ativa
# ---------------------------------------------------------------------------------------------


def _com_novidade(inf, valor: float):
    return dataclasses.replace(inf, tiles=tuple(dataclasses.replace(t, novidade=valor)
                                                for t in inf.tiles))


def test_fora_da_distribuicao_vale_mesmo_com_confianca(tmp_path, relogio) -> None:
    m = motor(tmp_path)
    quadro, r = primeira(m, cena(relogio, cor=(255, 0, 0)))  # mancha vermelha: evento firme
    cfg = ActivePerceptionConfig()
    assert r.category is not Category.NORMAL
    firme = assess(r, quadro, m, cfg)
    assert "fora_da_distribuicao" not in firme.reasons
    assert caso_de_inspecao(r, firme) == "anomalia_confiante"
    fora = assess(_com_novidade(r, 3.0), quadro, m, cfg)
    assert "fora_da_distribuicao" in fora.reasons and fora.decision_doubt
    assert caso_de_inspecao(r, fora) == "informacao_insuficiente"
    desligado = ActivePerceptionConfig(incerto_em_novidade=False)
    assert "fora_da_distribuicao" not in assess(_com_novidade(r, 3.0), quadro, m,
                                               desligado).reasons


def test_os_quatro_casos(tmp_path, relogio) -> None:
    m = motor(tmp_path)
    quadro, sadio = primeira(m, cena(relogio, lado=0))
    cfg = ActivePerceptionConfig()
    assert caso_de_inspecao(sadio, assess(sadio, quadro, m, cfg)) == "suficiente"
    assert caso_de_inspecao(sadio, Uncertainty(("evidencia_na_fronteira",), None)) == "ambiguo"
    assert caso_de_inspecao(sadio, Uncertainty(("qualidade_baixa",), None)) == (
        "informacao_insuficiente")


def test_termica_e_evidencia_e_nao_decide(tmp_path, relogio) -> None:
    m = motor(tmp_path)
    camera = cena(relogio, cor=AMBIGUA)
    cfg = ActivePerceptionConfig()
    sem = ActivePerception(cfg, relogio).run(m, camera, *primeira(m, camera))
    quente = SimulatedThermal(quente=(0.5, 0.5, 4.0), raio_px=25)
    ap = ActivePerception(cfg, relogio)
    camera = cena(relogio, cor=AMBIGUA)
    com = ap.run(m, camera, *primeira(m, camera), termica=quente)
    t = com.block["termica"]
    assert quente.capturas == 1 and t["calibrado"] is False and t["delta_c"] > 0
    assert com.block["caso"] in ("ambiguo", "informacao_insuficiente")
    assert com.inference.category is sem.inference.category  # a térmica não muda a decisão
    assert ap.snapshot()["termica"] == 1

    class Quebrada:
        def capture(self):
            raise RuntimeError("driver fora")

    camera = cena(relogio, cor=AMBIGUA)
    d = ActivePerception(cfg, relogio).run(m, camera, *primeira(m, camera), termica=Quebrada())
    assert d.block["termica"] is None and d.block["views"]  # segue sem ela


def test_medida_termica_na_regiao() -> None:
    t = SimulatedThermal(base_c=25.0, quente=(0.25, 0.25, 6.0))
    medida = medida_termica(t, (0.15, 0.15, 0.35, 0.35))
    assert medida["regiao_c"] > medida["resto_c"] == pytest.approx(25.0)
    assert json.dumps(medida)  # vai no corpo do resultado


def test_folha_nao_vista_perto_do_limiar_nao_dispara(tmp_path, relogio) -> None:
    """Folha de café que a cabeça não viu chega a 1,1 pelo caminho do quadro (medido): o
    limiar padrão (1,2) não a trata como fora da distribuição."""
    m = motor(tmp_path)
    quadro, r = primeira(m, cena(relogio, cor=(255, 0, 0)))
    cfg = ActivePerceptionConfig()
    assert cfg.limiar_novidade == pytest.approx(1.2)
    assert "fora_da_distribuicao" not in assess(_com_novidade(r, 1.1), quadro, m, cfg).reasons
    assert "fora_da_distribuicao" in assess(_com_novidade(r, 1.3), quadro, m, cfg).reasons
