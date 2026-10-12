from __future__ import annotations

import pytest
from conftest import HttpResponse, write_images

from valtexa_rover.hal.interfaces import (
    Camera,
    Gps,
    HardwareUnavailableError,
    Network,
    NetworkUnavailableError,
)
from valtexa_rover.hal.local import LocalDiskStorage, OnnxCpuAccelerator
from valtexa_rover.hal.simulated import (
    FlakyNetwork,
    FolderCamera,
    SimulatedController,
    SimulatedSensors,
    TrajectoryGps,
)


def test_camera_de_pasta_e_simulacao_marcada(tmp_path, clock) -> None:
    pasta = write_images(tmp_path / "img", ["verde", "vermelho"])
    cam = FolderCamera("cam-0", pasta, clock, loop=True, fail_on=[2])
    assert isinstance(cam, Camera)
    assert cam.describe().simulated is True
    q0, q1 = cam.capture(), cam.capture()
    assert (q0.frame_index, q1.frame_index) == (0, 1)
    assert q0.source.endswith("verde.jpg") and q0.content_type == "image/jpeg"
    assert (q0.width, q0.height) == (160, 120)
    with pytest.raises(HardwareUnavailableError):
        cam.capture()  # índice 2: falha programada
    assert cam.capture().source.endswith("vermelho.jpg")  # 3 → volta ao início (loop)


def test_camera_sem_loop_para_quando_acaba(tmp_path, clock) -> None:
    cam = FolderCamera("cam-0", write_images(tmp_path / "img", ["verde"]), clock, loop=False)
    cam.capture()
    with pytest.raises(HardwareUnavailableError):
        cam.capture()


def test_gps_percorre_a_trajetoria_e_perde_sinal(clock) -> None:
    gps = TrajectoryGps([(-45.0, -21.0), (-45.0, -20.99)], clock, speed_ms=10,
                        lost_between=(5, 6))
    assert isinstance(gps, Gps)
    inicio = gps.read()
    clock.advance(4)
    depois = gps.read()
    assert depois.lat > inicio.lat  # andou para o norte
    assert depois.heading_deg == pytest.approx(0.0, abs=1e-6)
    distancia = (depois.lat - inicio.lat) * 110_540
    assert distancia == pytest.approx(40, rel=0.01)
    clock.advance(1.5)
    with pytest.raises(HardwareUnavailableError):
        gps.read()
    clock.advance(1)
    assert gps.read().fix == "fixed"
    assert gps.describe().simulated


def test_rede_que_cai_e_volta() -> None:
    chamadas = []
    rede = FlakyNetwork(lambda m, p, b, h: chamadas.append(p) or HttpResponse(200, b"{}"))
    assert isinstance(rede, Network)
    rede.drop_after(2)
    rede.send("GET", "/a", b"", {})
    rede.send("GET", "/b", b"", {})
    with pytest.raises(NetworkUnavailableError):
        rede.send("GET", "/c", b"", {})
    assert not rede.is_online()
    rede.set_online(True)
    rede.send("GET", "/d", b"", {})
    assert chamadas == ["/a", "/b", "/d"] and rede.refused == 1
    assert rede.describe().simulated


def test_disco_local_com_cota_e_acelerador_reais(tmp_path) -> None:
    disco = LocalDiskStorage(tmp_path / "d", quota_bytes=10_000)
    (tmp_path / "d" / "x.bin").write_bytes(b"0" * 4000)
    total, livre = disco.usage()
    assert total == 10_000 and livre == 6000
    assert disco.describe().simulated is False
    acel = OnnxCpuAccelerator()
    assert "CPUExecutionProvider" in acel.providers()
    assert acel.describe().simulated is False


def test_sensores_e_controlador_simulados(clock) -> None:
    sensores = SimulatedSensors(clock, temperature_c=30, warming_c_per_h=2)
    clock.advance(3600)
    assert sensores.read().values["temperature_c"] == pytest.approx(32)
    ctl = SimulatedController()
    ctl.stop("teste")
    assert ctl.state()["stops"] == 1 and ctl.describe().simulated
