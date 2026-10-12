"""A câmera de folhas reais do simulador e a instalação de um modelo no formato do OTA."""

from __future__ import annotations

import hashlib
import io
from datetime import datetime

from PIL import Image

from valtexa_rover._compat import UTC
from valtexa_rover.clock import ManualClock
from valtexa_rover.hal.simulated import TrajectoryGps
from valtexa_rover.sim import CampoSimulado
from valtexa_rover.sim.folhas import CameraDeFolhas, instalar_modelo

CORES = {"Healthy": (20, 200, 20), "Rust": (220, 120, 20)}


def _folhas(pasta):
    for classe, cor in CORES.items():
        (pasta / classe).mkdir(parents=True)
        for i in range(3):
            Image.new("RGB", (400, 200), cor).save(pasta / classe / f"{i}.jpg")
    return pasta


def _camera(tmp_path, condicao: str, incidencia: float):
    campo = CampoSimulado.gerar(-47.06, -22.9, linhas=1, plantas_por_linha=2,
                                doentes={"L1-P001": condicao, "L1-P002": condicao})
    relogio = ManualClock(datetime(2026, 10, 10, tzinfo=UTC))
    gps = TrajectoryGps(campo.trajetoria(), relogio)
    return CameraDeFolhas("cam-0", campo, gps, relogio, _folhas(tmp_path / "folhas"),
                          lado="right", altura_mm=1000.0, incidencia=incidencia)


def _cor_do_centro(frame) -> tuple[int, int, int]:
    im = Image.open(io.BytesIO(frame.data)).convert("RGB")
    return im.getpixel((im.width // 2, im.height // 2))


def test_um_recorte_por_quadro_com_a_geometria_do_quadro(tmp_path) -> None:
    cam = _camera(tmp_path, "sadia", 0.5)
    f = cam.capture()
    assert len(cam.recortes) == 1  # sem emenda entre fotos dentro do recorte
    assert (f.width, f.height) == cam.size == (224, 224)
    assert f.side == "right" and f.camera_height_mm == 1000.0 and f.distance_mm == 270.0
    assert 30 < f.hfov_deg < 32  # 150 mm a 270 mm
    assert f.source.startswith("simulador-folhas:L1-P")


def test_planta_sadia_so_tem_folha_sadia_e_doente_segue_a_incidencia(tmp_path) -> None:
    sadia = _camera(tmp_path / "a", "sadia", 1.0)
    assert all(_cor_do_centro(sadia.capture())[1] > 150 for _ in range(5))
    doente = _camera(tmp_path / "b", "doenca", 1.0)
    assert all(_cor_do_centro(doente.capture())[0] > 150 for _ in range(5))


def test_instalar_modelo_no_formato_do_ota(tmp_path) -> None:
    origem = tmp_path / "m" / "cafe.onnx"
    origem.parent.mkdir()
    origem.write_bytes(b"onnx")
    origem.with_suffix(".edge.json").write_text("{}", encoding="utf-8")
    sha = instalar_modelo(origem, tmp_path / "artefatos")
    assert sha == hashlib.sha256(b"onnx").hexdigest()
    assert (tmp_path / "artefatos" / f"{sha}.onnx").read_bytes() == b"onnx"
    assert (tmp_path / "artefatos" / f"{sha}.edge.json").exists()
