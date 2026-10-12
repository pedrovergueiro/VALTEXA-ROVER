"""Adapters do Rover V3, com dados no formato real (sem o hardware).

O SDK falso imita a API do ``depthai`` (camelCase de propósito).
"""

# ruff: noqa: N802

from __future__ import annotations

import io
import json
import struct
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from valtexa_rover._compat import UTC
from valtexa_rover.hal.interfaces import Frame, HardwareUnavailableError
from valtexa_rover.hal.rover_v3 import (
    JetsonAccelerator,
    NavegacaoPorArquivo,
    NmeaGnss,
    NmeaParser,
    OakDCamera,
    XsensCanImu,
    decode_xsens_int16x3,
    depth_median_mm,
    hal_rover_v3,
    parse_tegrastats,
)

AGORA = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)
# Sentenças clássicas do padrão NMEA 0183 (checksum válido).
GGA = "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47"
RMC = "$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A"


def _nmea(corpo: str) -> str:
    soma = 0
    for ch in corpo:
        soma ^= ord(ch)
    return f"${corpo}*{soma:02X}"


class Relogio:
    def __init__(self, agora: datetime) -> None:
        self.agora = agora

    def now(self) -> datetime:
        return self.agora

    def monotonic(self) -> float:
        return 0.0


class TestNmea:
    def test_gga_e_rmc(self) -> None:
        p = NmeaParser()
        assert p.feed(GGA, AGORA) and p.feed(RMC, AGORA)
        e = p.estado
        assert e.lat == pytest.approx(48 + 7.038 / 60)
        assert e.lon == pytest.approx(11 + 31.0 / 60)
        assert e.fix == "gps" and e.hdop == 0.9 and e.alt_m == 545.4
        assert e.speed_ms == pytest.approx(22.4 * 0.514444)
        assert e.heading_deg == 84.4

    def test_rtk_fixo_e_flutuante_e_hemisferio_sul(self) -> None:
        p = NmeaParser()
        assert p.feed(_nmea("GNGGA,120000,2254.000,S,04703.600,W,4,20,0.5,600.0,M,0,M,1,0000"),
                      AGORA)
        assert p.estado.fix == "fixed"
        assert p.estado.lat == pytest.approx(-(22 + 54 / 60))
        assert p.estado.lon == pytest.approx(-(47 + 3.6 / 60))
        p.feed(_nmea("GNGGA,120001,2254.000,S,04703.600,W,5,20,0.5,600.0,M,0,M,1,0000"), AGORA)
        assert p.estado.fix == "float"

    def test_checksum_errado_e_descartado(self) -> None:
        p = NmeaParser()
        assert not p.feed(GGA[:-2] + "00", AGORA)
        assert p.estado.lat is None

    def test_adapter_le_da_serial_e_recusa_fix_velho(self) -> None:
        relogio = Relogio(AGORA)
        gnss = NmeaGnss("/dev/ttyACM0", clock=relogio, stream=io.StringIO(GGA + "\n" + RMC + "\n"))
        for _ in range(100):  # a thread lê o "arquivo" na hora
            try:
                fix = gnss.read()
                break
            except HardwareUnavailableError:
                import time

                time.sleep(0.01)
        assert fix.fix == "gps" and fix.heading_deg == 84.4
        relogio.agora = AGORA + timedelta(seconds=10)
        with pytest.raises(HardwareUnavailableError, match=r"há 10 s"):
            gnss.read()


def test_tegrastats_orin() -> None:
    linha = ("10-07-2026 12:00:00 RAM 2448/15388MB (lfb 2x4MB) SWAP 0/7694MB (cached 0MB) "
             "CPU [3%@729,1%@729,0%@729,2%@729,off,off,off,off] EMC_FREQ 0%@2133 "
             "GR3D_FREQ 37%@[611,0] cpu@44.25C soc2@42.5C gpu@43.875C tj@44.25C")
    t = parse_tegrastats(linha)
    assert t["accelerator_percent"] == 37.0
    assert t["accelerator_temperature_c"] == 43.875
    assert t["memory_used_bytes"] == 2448 * 1024 * 1024
    assert t["cpu_percent"] == pytest.approx(1.5)
    assert parse_tegrastats("linha sem nada")["accelerator_percent"] is None


def test_acelerador_da_jetson_cai_para_cpu_fora_dela() -> None:
    provedores = JetsonAccelerator().providers()
    assert "CPUExecutionProvider" in provedores
    assert JetsonAccelerator().describe().simulated is False


class TestXsens:
    def test_decodifica_int16_big_endian_com_escala(self) -> None:
        dados = struct.pack(">hhh", 128 * 10, -128 * 5, 128 * 90)
        assert decode_xsens_int16x3(dados, 2**-7) == (10.0, -5.0, 90.0)
        with pytest.raises(ValueError):
            decode_xsens_int16x3(b"\x00\x01", 1.0)

    def test_adapter_guarda_ultima_leitura(self) -> None:
        class Barramento:
            def recv(self, timeout):  # sem mensagens: a thread só espera
                import time

                time.sleep(timeout)

        relogio = Relogio(AGORA)
        imu = XsensCanImu(clock=relogio, bus=Barramento())
        with pytest.raises(HardwareUnavailableError):
            imu.read()
        imu.handle(0x22, struct.pack(">hhh", 128, 256, -384))
        imu.handle(0x34, struct.pack(">hhh", 0, 0, 256 * 9))
        leitura = imu.read()
        assert (leitura.roll_deg, leitura.pitch_deg, leitura.yaw_deg) == (1.0, 2.0, -3.0)
        assert leitura.accel_ms2 == (0.0, 0.0, 9.0)


class TestOakD:
    def test_mediana_da_profundidade_no_centro_sem_zeros(self) -> None:
        prof = np.zeros((400, 640), dtype=np.uint16)
        prof[100:300, 160:480] = 270
        prof[150:160, 200:220] = 0  # buracos sem medida
        prof[0:50, :] = 9000  # céu na borda: fora do centro
        assert depth_median_mm(prof) == 270.0
        assert depth_median_mm(np.zeros((400, 640), np.uint16)) is None

    def test_captura_com_sdk_falso(self) -> None:
        """O que o adapter faz com o que o SDK devolve (o SDK real não roda aqui)."""
        jpeg = io.BytesIO()
        Image.new("RGB", (4056, 3040), (40, 120, 40)).save(jpeg, "JPEG")
        prof = np.full((480, 640), 300, dtype=np.uint16)

        class No:
            def __getattr__(self, _):
                return self

            def __call__(self, *a, **k):
                return self

            def link(self, *_):
                pass

        class Fila:
            def __init__(self, item=None):
                self.item = item
                self.enviados = []

            def get(self):
                return self.item

            def tryGet(self):
                return self.item

            def send(self, x):
                self.enviados.append(x)

        class Pacote:
            def __init__(self, dados=None, quadro=None):
                self.dados, self.quadro = dados, quadro

            def getData(self):
                return np.frombuffer(self.dados, dtype=np.uint8)

            def getFrame(self):
                return self.quadro

        class Dispositivo:
            def __init__(self, *_):
                self.filas = {"still": Fila(Pacote(dados=jpeg.getvalue())),
                              "depth": Fila(Pacote(quadro=prof)), "control": Fila()}

            def getOutputQueue(self, nome, **_):
                return self.filas[nome]

            def getInputQueue(self, nome):
                return self.filas[nome]

        class Dai(No):
            Device = Dispositivo

            class CameraControl:
                def setCaptureStill(self, v):
                    self.still = v

        cam = OakDCamera("esq-1000", "192.168.10.12", 1000.0, clock=Relogio(AGORA), dai=Dai())
        quadro = cam.capture()
        assert isinstance(quadro, Frame)
        assert (quadro.width, quadro.height) == (4056, 3040)
        assert quadro.distance_mm == 300.0 and quadro.camera_height_mm == 1000.0
        assert quadro.content_type == "image/jpeg" and quadro.frame_index == 1
        assert quadro.side is None
        with pytest.raises(ValueError, match="lado"):
            OakDCamera("x", "1.2.3.4", 450.0, dai=Dai(), side="esquerda")


def test_monta_o_hardware_pelo_json(tmp_path: Path) -> None:
    exemplo = Path(__file__).resolve().parents[1] / "deploy" / "hardware.rover-v3.json.exemplo"
    (tmp_path / "hardware.json").write_text(exemplo.read_text(encoding="utf-8"), encoding="utf-8")
    feitas = {"cameras": []}

    def camera(c):
        feitas["cameras"].append(c["id"])
        return type("C", (), {"camera_id": c["id"], "describe": lambda s: None})()

    hw = hal_rover_v3(dados=tmp_path, url="https://exemplo", clock=Relogio(AGORA),
                      camera=camera, gnss=lambda g: g["porta"], imu=lambda i: i["canal"])
    assert len(feitas["cameras"]) == 6 and hw.gps == "/dev/valtexa-gnss" and hw.imu == "can0"
    assert isinstance(hw.controller, NavegacaoPorArquivo)
    assert hw.controller.missao == Path("/run/valtexa/missao.json")
    assert hw.hardware_version == "Rover V3 rev E"
    assert json.loads(exemplo.read_text(encoding="utf-8"))["imu"]["ids"]["euler_id"] == 0x22


def test_sem_hardware_json_explica(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("VALTEXA_HARDWARE", raising=False)
    with pytest.raises(HardwareUnavailableError, match=r"hardware\.rover-v3\.json\.exemplo"):
        hal_rover_v3(dados=tmp_path, url="https://exemplo", clock=Relogio(AGORA))


class TestNavegacao:
    def test_estado_por_arquivo(self, tmp_path: Path) -> None:
        relogio = Relogio(AGORA)
        nav = NavegacaoPorArquivo(tmp_path / "navegacao.json", tmp_path / "parar.json",
                                  clock=relogio)
        assert nav.state() == {"route_done": False, "navegacao": "sem estado"}
        (tmp_path / "navegacao.json").write_text('{"mission_id": "m-1", "route_done": tr')
        assert nav.state()["route_done"] is False  # escrita pela metade
        (tmp_path / "navegacao.json").write_text('{"mission_id": "m-1", "route_done": true}')
        assert nav.state() == {"mission_id": "m-1", "route_done": True}
        (tmp_path / "navegacao.json").write_text("[1, 2]")
        assert nav.state()["route_done"] is False

    def test_estado_velho_nao_conclui(self, tmp_path: Path) -> None:
        estado = tmp_path / "navegacao.json"
        estado.write_text('{"route_done": true}')
        quando = estado.stat().st_mtime
        relogio = Relogio(datetime.fromtimestamp(quando + 60, UTC))
        nav = NavegacaoPorArquivo(estado, tmp_path / "parar.json", clock=relogio, validade_s=10)
        assert nav.state()["route_done"] is False and "velho" in nav.state()["navegacao"]

    def test_parada_e_missao_anunciada(self, tmp_path: Path) -> None:
        nav = NavegacaoPorArquivo(tmp_path / "run" / "navegacao.json",
                                  tmp_path / "run" / "parar.json", clock=Relogio(AGORA))
        nav.stop("disco cheio")
        assert json.loads((tmp_path / "run" / "parar.json").read_text())["reason"] == "disco cheio"
        missao = type("M", (), {"id": "m-1", "field_id": "t-1", "payload": {"rota": [1]}})()
        nav.announce_mission(missao)
        anunciada = json.loads((tmp_path / "run" / "missao.json").read_text(encoding="utf-8"))
        assert anunciada["id"] == "m-1" and anunciada["payload"] == {"rota": [1]}
        nav.announce_mission(None)
        assert json.loads((tmp_path / "run" / "missao.json").read_text())["id"] is None
        assert not list((tmp_path / "run").glob("*.tmp"))
