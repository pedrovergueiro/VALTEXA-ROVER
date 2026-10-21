"""Fusão RGB + profundidade: a planta pelo pico mais próximo, o fundo fora da evidência.

A cena é a da :class:`~valtexa_rover.hal.simulated.SimulatedRgbdCamera` — a saia verde a
27 cm, vãos com o fundo a 2,5 m — e o modelo é o minúsculo do ``conftest`` (cor → classe).
O caso que motiva o módulo: um objeto vermelho **na rua**, visto por um vão do dossel. Sem
a profundidade, o recorte dele vira ferrugem; com ela, é fundo e sai da evidência.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from conftest import build_tiny_onnx
from test_edge_perfil import PERFIL

from valtexa_rover._compat import UTC
from valtexa_rover.clock import ManualClock
from valtexa_rover.config import (
    Category,
    ClassificationConfig,
    ConfigError,
    DepthConfig,
    RuntimeConfig,
)
from valtexa_rover.edge_ai import rgbd
from valtexa_rover.edge_ai.engine import EdgeAIEngine, ModelPackage
from valtexa_rover.hal.lente import LensModel, Undistorter
from valtexa_rover.hal.local import OnnxCpuAccelerator
from valtexa_rover.hal.simulated import SimulatedRgbdCamera

CFG = DepthConfig()
VERMELHO = (255, 0, 0)


@pytest.fixture
def relogio() -> ManualClock:
    return ManualClock(datetime(2026, 10, 15, 12, 0, tzinfo=UTC))


def motor(tmp_path: Path, depth: DepthConfig | None = None) -> EdgeAIEngine:
    caminho = tmp_path / "m.onnx"
    sha = build_tiny_onnx(caminho)
    caminho.with_suffix(".edge.json").write_text(json.dumps(PERFIL), encoding="utf-8")
    m = EdgeAIEngine(OnnxCpuAccelerator(), ClassificationConfig(), depth)
    m.load(ModelPackage("coffee-foliar", "1", sha, caminho))
    return m


class TestPlanoDaPlanta:
    def test_pico_mais_proximo_e_nao_a_mediana(self) -> None:
        prof = np.full((240, 320), 270.0)
        prof[:, 90:320] = 2500.0  # 86% do centro é fundo visto por um vão largo
        plano = rgbd.plant_plane(prof, CFG)
        assert plano.sufficient and plano.distance_mm == pytest.approx(270.0, abs=13)
        assert plano.near_mm == pytest.approx(120.0, abs=13)
        assert plano.far_mm == pytest.approx(720.0, abs=13)
        centro = prof[48:192, 64:256]
        assert float(np.median(centro)) == 2500.0  # a mediana do quadro erraria a escala

    def test_folha_solta_na_frente_nao_e_a_planta(self) -> None:
        prof = np.full((240, 320), 400.0)
        prof[110:118, 150:160] = 150.0  # uma folha a 15 cm, pouco do quadro
        assert rgbd.plant_plane(prof, CFG).distance_mm == pytest.approx(400.0, abs=13)

    def test_sem_medida_suficiente_nao_decide(self) -> None:
        prof = np.zeros((240, 320))
        prof[::10, ::10] = 300.0  # 1% medido
        plano = rgbd.plant_plane(prof, CFG)
        assert not plano.sufficient and plano.distance_mm is None
        assert plano.coverage == pytest.approx(0.01, abs=0.005)
        assert not rgbd.plant_plane(None, CFG).sufficient

    def test_recorte_sem_medida_fica_e_recorte_de_fundo_sai(self) -> None:
        prof = np.full((240, 320), 270.0)
        prof[:, 200:] = 2500.0
        prof[:, :40] = 0.0
        plano = rgbd.plant_plane(prof, CFG)
        caixas = [(0, 0, 200, 200), (500, 0, 700, 200), (1100, 0, 1300, 200)]
        sem_medida, planta, fundo = rgbd.tile_depths(prof, (1600, 1200), caixas, plano, CFG)
        assert sem_medida.is_plant is None and sem_medida.valid == 0.0
        assert planta.is_plant is True and planta.depth_mm == 270.0
        assert fundo.is_plant is False and fundo.behind == 1.0

    def test_abertura_do_dossel_por_terco(self) -> None:
        plano = rgbd.PlantPlane(270.0, 120.0, 720.0, 1.0, True)
        tiles = [rgbd.TileDepth(270.0, 1.0, 1.0, 0.0, True),
                 rgbd.TileDepth(270.0, 1.0, 0.5, 0.5, True),
                 rgbd.TileDepth(2500.0, 1.0, 0.0, 1.0, False)]
        resumo = rgbd.summary(plano, tiles, [["terco_inferior"], ["terco_superior"],
                                             ["terco_superior"]], "filtrar", 1)
        assert resumo["canopy_openness"] == pytest.approx(0.5)
        assert resumo["canopy_openness_by_zone"] == {"terco_inferior": 0.0,
                                                     "terco_superior": 0.75}
        assert resumo["calibrated"] is False and resumo["excluded_tiles"] == 1


class TestMotorComProfundidade:
    def _cena(self, relogio, **kw) -> SimulatedRgbdCamera:
        return SimulatedRgbdCamera("c", relogio, vaos=[(1100, 1400)], manchas=[
            {"x": 1120, "y": 400, "lado": 260, "cor": VERMELHO, "no_fundo": True}], **kw)

    def test_objeto_vermelho_no_vao_nao_vira_ferrugem(self, tmp_path, relogio) -> None:
        quadro = self._cena(relogio).capture()
        sem = motor(tmp_path / "a", DepthConfig(modo="desligado"))
        r = sem.infer(quadro.data, quadro.distance_mm, 1000.0, depth=quadro.depth)
        assert r.category is Category.POSSIVEL_DOENCA and r.rgbd is None  # o alarme falso

        com = motor(tmp_path / "b")
        r = com.infer(quadro.data, quadro.distance_mm, 1000.0, depth=quadro.depth)
        assert r.category is Category.NORMAL
        assert r.excluded_tiles > 0 and r.rgbd["excluded_tiles"] == r.excluded_tiles
        assert r.rgbd["plant_distance_mm"] == pytest.approx(270.0, abs=13)
        assert r.rgbd["canopy_openness"] > 0.1  # o vão aparece como abertura do dossel
        assert all(t.depth_mm is not None and t.plant_fraction >= 0.35 for t in r.tiles)

    def test_anotar_mede_mas_nao_tira_recorte(self, tmp_path, relogio) -> None:
        quadro = self._cena(relogio).capture()
        r = motor(tmp_path, DepthConfig(modo="anotar")).infer(
            quadro.data, quadro.distance_mm, 1000.0, depth=quadro.depth)
        assert r.category is Category.POSSIVEL_DOENCA and r.excluded_tiles == 0
        assert r.rgbd["mode"] == "anotar" and r.rgbd["sufficient"]

    def test_profundidade_ruim_nao_filtra_nada(self, tmp_path, relogio) -> None:
        quadro = self._cena(relogio, cobertura=0.05).capture()
        r = motor(tmp_path).infer(quadro.data, quadro.distance_mm, 1000.0, depth=quadro.depth)
        assert r.rgbd["sufficient"] is False and r.excluded_tiles == 0
        assert not r.distance_measured or r.distance_mm == quadro.distance_mm

    def test_distancia_pelo_pico_muda_o_tamanho_do_recorte(self, tmp_path, relogio) -> None:
        cena = SimulatedRgbdCamera("c", relogio, distancia_mm=400.0, vaos=[(0, 1000)])
        quadro = cena.capture()
        assert quadro.distance_mm == 2500.0  # a mediana: o vão ocupa o centro
        r = motor(tmp_path).infer(quadro.data, quadro.distance_mm, 1000.0, depth=quadro.depth)
        assert r.distance_mm == pytest.approx(400.0, abs=13) and r.distance_measured

    def test_regiao_so_de_fundo_sai_sem_recorte_e_sem_baixa_confianca(self, tmp_path,
                                                                      relogio) -> None:
        cena = SimulatedRgbdCamera("c", relogio, vaos=[(0, 300)])
        quadro = cena.capture()
        r = motor(tmp_path).infer(quadro.data, quadro.distance_mm, 1000.0, depth=quadro.depth,
                                  region=(0.0, 0.0, 0.15, 1.0))
        # Vão entre plantas: nada a ler, e não é dúvida do modelo (não sobe original).
        assert r.tiles == () and r.category is Category.NORMAL and not r.low_confidence
        assert "faixa de profundidade" in r.reason and r.rgbd["excluded_tiles"] > 0


def test_lente_corrige_a_profundidade_sem_interpolar() -> None:
    lente = LensModel(320, 240, 160.0, 160.0, 160.0, 120.0, "perspective", (-0.3, 0.1))
    prof = np.full((240, 320), 270, dtype=np.uint16)
    prof[:, 160:] = 2500
    corrigida = Undistorter(lente).apply_depth(prof)
    assert set(np.unique(corrigida)) <= {0, 270, 2500}  # nenhum valor intermediário
    assert corrigida[120, 100] == 270 and corrigida[120, 220] == 2500


def test_configuracao_de_profundidade_e_validada(tmp_path) -> None:
    caminho = tmp_path / "rover.json"
    caminho.write_text(json.dumps({"profundidade": {"modo": "talvez"}}))
    with pytest.raises(ConfigError, match=r"profundidade.modo"):
        RuntimeConfig.load(caminho)
    caminho.write_text(json.dumps({"profundidade": {"min_na_planta": 2}}))
    with pytest.raises(ConfigError, match="min_na_planta"):
        RuntimeConfig.load(caminho)
