"""Correção da lente: a foto vira a de uma câmera ideal, e a escala dos recortes fica exata."""

from __future__ import annotations

import io
import json
import math

import numpy as np
import pytest
from PIL import Image

from valtexa_rover.hal.lente import LensModel, Undistorter

W, H = 400, 300


def grade(lente: LensModel) -> np.ndarray:
    """Pontos de uma grade regular (câmera ideal) desenhados onde a LENTE os põe."""
    img = np.zeros((H, W, 3), np.uint8)
    for gx in range(40, W, 40):
        for gy in range(30, H, 30):
            x, y = (gx - lente.cx) / lente.fx, (gy - lente.cy) / lente.fy
            xd, yd = lente.distort(np.array(x), np.array(y))
            u, v = round(float(xd) * lente.fx + lente.cx), round(float(yd) * lente.fy
                                                                          + lente.cy)
            if 1 <= u < W - 1 and 1 <= v < H - 1:
                img[v - 1:v + 2, u - 1:u + 2] = 255
    return img


@pytest.mark.parametrize(("modelo", "coefs"), [
    ("perspective", (-0.25, 0.08, 0.001, -0.001, -0.01)),
    ("fisheye", (-0.02, 0.01, -0.004, 0.001)),
])
def test_correcao_devolve_a_grade_ao_lugar(modelo, coefs) -> None:
    lente = LensModel(W, H, 180.0, 180.0, W / 2, H / 2, modelo, coefs)
    corrigida = Undistorter(lente).apply(grade(lente))
    erros = []
    for gx in range(80, W - 60, 40):
        for gy in range(60, H - 40, 30):
            janela = corrigida[gy - 4:gy + 5, gx - 4:gx + 5, 0].astype(float)
            if janela.sum() == 0:
                continue
            ys, xs = np.mgrid[0:9, 0:9]
            vx, vy = (janela * xs).sum() / janela.sum(), (janela * ys).sum() / janela.sum()
            erros.append(math.hypot(vx - 4, vy - 4))
    assert erros and max(erros) <= 1.0  # cada ponto volta para onde a câmera ideal o poria


def test_campo_de_visao_efetivo_e_escala() -> None:
    lente = LensModel(4056, 3040, 1015.0, 1015.0, 2028, 1520, "fisheye", (0.0, 0.0, 0.0, 0.0))
    u = Undistorter(lente)
    assert u.hfov_deg == pytest.approx(math.degrees(2 * math.atan(4056 / 2030)), abs=1e-6)
    # mm por pixel a 270 mm pela fórmula do recorte = distância / fx
    gsd = 2 * 270 * math.tan(math.radians(u.hfov_deg) / 2) / 4056
    assert gsd == pytest.approx(270 / 1015.0, rel=1e-9)


def test_sem_distorcao_nao_muda_a_foto() -> None:
    rng = np.random.default_rng(0)
    foto = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    lente = LensModel(W, H, 200.0, 200.0, W / 2 - 0.5, H / 2 - 0.5)
    assert np.abs(Undistorter(lente).apply(foto).astype(int) - foto).max() <= 1


def test_oakd_corrige_e_manda_o_campo_de_visao(monkeypatch) -> None:
    """Com a calibração do aparelho, a foto sai corrigida e com o HFOV efetivo."""
    from test_rover_v3 import AGORA, Relogio

    from valtexa_rover.hal.rover_v3 import OakDCamera

    jpeg = io.BytesIO()
    Image.new("RGB", (4056, 3040), (40, 120, 40)).save(jpeg, "JPEG")

    class Calib:
        def getCameraIntrinsics(self, _s, w, h):  # noqa: N802 — API do depthai
            return [[1015.0, 0, w / 2], [0, 1015.0, h / 2], [0, 0, 1]]

        def getDistortionCoefficients(self, _s):  # noqa: N802
            return [-0.02, 0.01, -0.004, 0.001] + [0.0] * 10

        def getDistortionModel(self, _s):  # noqa: N802
            return "CameraModel.Fisheye"

    def abrir(self, fps):
        self._device = type("D", (), {"readCalibration": lambda s: Calib()})()

    monkeypatch.setattr(OakDCamera, "_abrir", abrir)
    dai = type("Dai", (), {"CameraBoardSocket": type("S", (), {"CAM_A": 0})})()
    cam = OakDCamera("esq-1000", "192.168.10.12", 1000.0, clock=Relogio(AGORA), dai=dai)
    assert cam.undistorter is not None and cam.undistorter.lens.model == "fisheye"
    assert 120 < cam.undistorter.hfov_deg < 128
    assert "corrigida (fisheye" in json.dumps(cam.describe().detail)
