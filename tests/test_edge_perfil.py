"""Edge AI com perfil: recortes de tamanho físico fixo e evento por recorte.

O modelo é o minúsculo do ``conftest`` (cor dominante → classe), em 32 px. O quadro é um
campo verde com uma mancha vermelha: com recorte, a mancha vira evento; sem recorte, o
quadro inteiro reduzido a 32 px dilui a mancha e passa como sadio — o problema que o
recorte resolve no modelo real (uma lesão de Sigatoka num quadro de 1 m de planta).
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
import pytest
from conftest import build_tiny_onnx
from PIL import Image

from valtexa_rover.config import Category, ClassificationConfig
from valtexa_rover.edge_ai.engine import EdgeAIEngine, EdgeAIError, ModelPackage
from valtexa_rover.edge_ai.perfil import EdgeProfile, ProfileError
from valtexa_rover.hal.local import OnnxCpuAccelerator

PERFIL = {
    "version": 1,
    "crop": "banana",
    "healthy_label": "Healthy",
    "event_threshold": 0.8,
    "min_event_tiles": 1,
    "near_miss_threshold": 0.5,
    "anomaly_mean_confidence_below": None,
    "tile_mm": 150.0,
    "overlap": 0.25,
    "hfov_deg": 127.0,
    "vfov_deg": 110.0,
    "default_distance_mm": 270.0,
    "tile_batch": 8,
    "tile_probabilities": "full",
    "measured": {"teste": True},
}


def quadro(mancha: bool, size=(1600, 1200)) -> bytes:
    rgb = np.zeros((size[1], size[0], 3), np.uint8)
    rgb[..., 1] = 255
    if mancha:
        rgb[450:750, 650:950] = (255, 0, 0)
    saida = io.BytesIO()
    Image.fromarray(rgb).save(saida, "JPEG", quality=92)
    return saida.getvalue()


def motor(tmp_path: Path, perfil: dict | None = PERFIL) -> EdgeAIEngine:
    caminho = tmp_path / "m.onnx"
    sha = build_tiny_onnx(caminho)
    if perfil is not None:
        caminho.with_suffix(".edge.json").write_text(json.dumps(perfil), encoding="utf-8")
    m = EdgeAIEngine(OnnxCpuAccelerator(), ClassificationConfig())
    m.load(ModelPackage("banana-foliar", "1.0", sha, caminho))
    return m


def test_mancha_vira_evento_so_com_recorte(tmp_path) -> None:
    com = motor(tmp_path / "a")
    r = com.infer(quadro(True), distance_mm=270.0, camera_height_mm=1000.0)
    assert r.category is Category.POSSIVEL_DOENCA and r.label == "Rust"
    assert r.event_tiles >= 1 and r.tiles and r.distance_measured
    assert all(len(t.probabilities) == 4 for t in r.tiles)
    lado = r.tiles[0].box[2] - r.tiles[0].box[0]
    assert 200 < lado < 240  # 150 mm a 270 mm num quadro de 1600 px e 127°

    sem = motor(tmp_path / "b", perfil=None)
    assert sem.infer(quadro(True)).category is Category.NORMAL  # a mancha some no quadro inteiro


def test_quadro_sadio_e_zonas_pela_altura(tmp_path) -> None:
    r = motor(tmp_path).infer(quadro(False), distance_mm=270.0, camera_height_mm=1000.0)
    assert r.category is Category.NORMAL and not r.low_confidence and r.event_tiles == 0
    assert all("folhas_baixas" in t.zones or "pseudocaule" in t.zones for t in r.tiles)


def test_sem_distancia_usa_a_do_perfil(tmp_path) -> None:
    r = motor(tmp_path).infer(quadro(False))
    assert r.distance_mm == 270.0 and not r.distance_measured


def test_planta_longe_demais_guarda_o_original(tmp_path) -> None:
    r = motor(tmp_path).infer(quadro(True), distance_mm=5000.0, camera_height_mm=1000.0)
    assert r.category is Category.NORMAL and r.low_confidence and not r.tiles
    assert "sem recorte" in r.reason


def test_quase_evento_vai_para_validacao(tmp_path) -> None:
    alto = {**PERFIL, "event_threshold": 1.0, "near_miss_threshold": 0.5}  # inalcançável
    r = motor(tmp_path, alto).infer(quadro(True), distance_mm=270.0)
    assert r.category is Category.NORMAL and r.low_confidence
    assert "abaixo do evento" in r.reason


def test_perfil_incoerente_e_recusado(tmp_path) -> None:
    with pytest.raises(ProfileError):
        EdgeProfile.from_dict({**PERFIL, "event_threshold": 1.5})
    with pytest.raises(ProfileError):
        EdgeProfile.from_dict({**PERFIL, "near_miss_threshold": 0.9})
    with pytest.raises(ProfileError):
        EdgeProfile.from_dict({**PERFIL, "chave_que_nao_existe": 1})
    with pytest.raises(EdgeAIError, match="rótulo sadio"):
        motor(tmp_path, {**PERFIL, "healthy_label": "Saudavel"})


def test_recortes_vao_nos_metadados_para_a_nuvem(rig_factory) -> None:
    rig = rig_factory(["vermelho", "verde"])
    # O perfil chega depois do boot (como num OTA de modelo com o mesmo ONNX): a missão
    # tem de usá-lo sem reiniciar o robô.
    for pasta in (rig.repo, rig.core.models.models_dir):
        (pasta / f"{rig.sha}.edge.json").write_text(json.dumps(PERFIL), encoding="utf-8")
    # Quadros de 1.600 px: os de 160 px do rig são pequenos demais para recortar 150 mm.
    for arquivo, mancha in zip(sorted(rig.images.iterdir()), (True, False), strict=True):
        arquivo.write_bytes(quadro(mancha))
    rig.core.start_mission("m-1", "talhao-1")
    rig.run(2)
    resultados = [json.loads(r["payload"]) for r in rig.core.store.query(
        "SELECT payload FROM records WHERE kind = 'result' ORDER BY occurred_at")]
    assert resultados[0]["tiles"]["labels"] == ["Healthy", "Rust", "Leaf_Miner_Damage",
                                                "Unknown_Background"]
    for r in resultados:
        t = r["tiles"]
        assert len(t["probabilities"]) == len(t["boxes"]) == len(t["heights_mm"]) >= 1
        assert t["tile_mm"] == 150.0 and "event_tiles" in r["result"]
        assert t["hfov_deg"] == 127.0 and "side" in t and "camera_height_mm" in t
        assert "navigation" in r


MODELOS_REAIS = os.environ.get("VALTEXA_MODELOS_REAIS")


@pytest.mark.skipif(not MODELOS_REAIS, reason="defina VALTEXA_MODELOS_REAIS com os .onnx reais")
@pytest.mark.parametrize("chave", ["coffee_foliar", "banana_foliar", "citrus_foliar"])
def test_modelo_real_carrega_e_recorta(tmp_path, chave) -> None:
    """Com os pesos reais (fora do repositório): carrega, recorta e classifica um quadro."""
    import hashlib

    origem = Path(MODELOS_REAIS) / f"{chave}.onnx"
    if not origem.with_suffix(".edge.json").exists():
        pytest.skip(f"{chave}.edge.json ainda não calibrado")
    m = EdgeAIEngine(OnnxCpuAccelerator(), ClassificationConfig())
    m.load(ModelPackage(chave, "real", hashlib.sha256(origem.read_bytes()).hexdigest(), origem))
    r = m.infer(quadro(False, size=(4056, 3040)), distance_mm=270.0, camera_height_mm=1000.0)
    assert len(r.tiles) > 20 and r.category in set(Category)


def test_lote_fixo_para_tensorrt(tmp_path) -> None:
    """Com TensorRT o último lote é completado: uma forma só, um motor compilado só."""
    m = motor(tmp_path)
    livre = m.infer(quadro(True), distance_mm=270.0)
    formas = []
    original = m._model

    def gravando(lote, training=False):
        formas.append(lote.shape[0])
        return original(lote)

    m._model, m.fixed_batch = gravando, True
    fixo = m.infer(quadro(True), distance_mm=270.0)
    assert set(formas) == {PERFIL["tile_batch"]} and len(livre.tiles) % PERFIL["tile_batch"]
    assert [t.probabilities for t in fixo.tiles] == [t.probabilities for t in livre.tiles]
    assert fixo.category is livre.category and fixo.event_tiles == livre.event_tiles


def _evidencia(evento: float, quase: float | None = None, rho: float = 0.1) -> dict:
    """Rust a favor só acima de 0,9 (neutro abaixo); as outras sempre contra."""
    return {"bin_edges": [0.0, 0.5, 0.9, 1.0],
            "llr_tables": {"Rust": [0.0, 0.0, 2.0], "Leaf_Miner_Damage": [-0.5, -0.5, -0.5],
                           "Unknown_Background": [-1.0, -1.0, -1.0]},
            "llr_clip": 1.5, "tile_correlation": rho, "event_evidence": evento,
            "near_miss_evidence": quase}


def test_regra_por_evidencia_decide_o_quadro(tmp_path) -> None:
    # Um recorte vermelho (+1,5) entre ~70 neutros, rho 0,1: 1,5 / (1 + 0,1 x 69) ≈ 0,19.
    perfil = {**PERFIL, "evidence": _evidencia(0.1)}
    r = motor(tmp_path / "a", perfil).infer(quadro(True), distance_mm=270.0)
    assert r.category is Category.POSSIVEL_DOENCA and r.label == "Rust"
    assert r.evidence is not None and r.evidence > 0.1 and "evidência de Rust" in r.reason
    assert r.event_tiles >= 1

    sadio = motor(tmp_path / "b", perfil).infer(quadro(False), distance_mm=270.0)
    assert sadio.category is Category.NORMAL and sadio.evidence <= 0.0

    alto = {**PERFIL, "evidence": _evidencia(1e6, quase=0.0)}  # inalcançável; validação baixa
    r = motor(tmp_path / "c", alto).infer(quadro(True), distance_mm=270.0)
    assert r.category is Category.NORMAL and r.low_confidence and "abaixo do evento" in r.reason


def test_evidencia_incoerente_e_recusada(tmp_path) -> None:
    for ruim in ({**_evidencia(0.5), "bin_edges": [0.0, 0.9, 0.5, 1.0]},
                 {**_evidencia(0.5), "llr_tables": {"Rust": [1.0]}},
                 _evidencia(0.5, quase=0.9),
                 {**_evidencia(0.5), "extra": 1}):
        with pytest.raises(ProfileError):
            EdgeProfile.from_dict({**PERFIL, "evidence": ruim})
    outra = {**_evidencia(0.5), "llr_tables": {"Mancha": [0.0, 0.0, 0.0]}}
    with pytest.raises(EdgeAIError, match="Mancha"):
        motor(tmp_path, {**PERFIL, "evidence": outra})


def test_desconto_de_correlacao() -> None:
    from valtexa_rover.edge_ai.perfil import EvidenceRule

    soma = EvidenceRule.from_dict(_evidencia(0.5, rho=0.0))
    media = EvidenceRule.from_dict(_evidencia(0.5, rho=1.0))
    probs = [0.95] + [0.7] * 9  # um recorte a favor (+1,5 limitado), nove neutros (0,0)
    assert soma.score("Rust", probs) == pytest.approx(1.5)
    assert media.score("Rust", probs) == pytest.approx(0.15)
