"""Adapters do **Rover V3** (``VALTEXA-SAAS/hardware/rover-v3``, rev E).

Componentes do projeto e o adapter de cada um:

=============================  ==============================  =================================
Peça (BOM rev E)               Adapter                         Estado
=============================  ==============================  =================================
Jetson Orin NX 16 GB           :class:`JetsonAccelerator`,     providers do ONNX Runtime;
                               :class:`JetsonSystemMonitor`    ``tegrastats`` (GPU, temperatura)
8 x Luxonis OAK-D Pro W PoE    :class:`OakDCamera`             SDK ``depthai`` 2.x, por IP
u-blox ZED-F9P (RTK)           :class:`NmeaGnss`               NMEA (GGA/RMC) pela serial/USB
Xsens MTi-630 (CAN)            :class:`XsensCanImu`            ``python-can``; IDs e escalas
                                                               configuráveis
Microcontrolador de segurança  —                               protocolo não definido: sem
+ MPW86 / ZLTECH                                               adapter (o Runtime não comanda
                                                               movimento; ver ``interfaces``)
Livox Mid-360 + navegação      :class:`NavegacaoPorArquivo`    fim de rota e pedido de parada
                                                               por arquivo em ``/run/valtexa``
=============================  ==============================  =================================

**O que foi testado e o que não foi.** O parser NMEA, o de ``tegrastats``, a decodificação
das mensagens CAN, a profundidade da OAK-D e a montagem pelo ``hardware.json`` têm testes
com dados no formato real (``tests/test_rover_v3.py``). **Nenhum adapter rodou no
hardware**: o caminho do ``depthai`` (pipeline, fila de quadro parado) e o barramento CAN
real precisam ser conferidos na primeira montagem — o roteiro está em
``docs/ROVER_RUNTIME.md``, seção "Rover V3". Os IDs e escalas padrão da Xsens vêm da
documentação pública do *MT CAN protocol* e **devem ser conferidos** no MT Manager do
sensor instalado (são configuráveis no ``hardware.json``).

Montagem: ``valtexa-rover servico --hal valtexa_rover.hal.rover_v3:hal_rover_v3`` lê
``<dados>/hardware.json`` (ou ``VALTEXA_HARDWARE``); exemplo em
``deploy/hardware.rover-v3.json.exemplo``.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from valtexa_rover._compat import UTC
from valtexa_rover.clock import Clock
from valtexa_rover.hal.interfaces import (
    DeviceInfo,
    Frame,
    GpsFix,
    Hardware,
    HardwareUnavailableError,
    ImuReading,
    SystemStats,
)
from valtexa_rover.hal.local import (
    HttpNetwork,
    LinuxSystemMonitor,
    LocalDiskStorage,
    OnnxCpuAccelerator,
)

HARDWARE_VERSION = "Rover V3 rev E"


# ---------------------------------------------------------------------------------------------
# Computador embarcado: Jetson Orin NX 16 GB
# ---------------------------------------------------------------------------------------------


class JetsonAccelerator(OnnxCpuAccelerator):
    """TensorRT → CUDA → CPU, na ordem, entre os que o ONNX Runtime instalado oferece.

    Na Jetson o ONNX Runtime vem do índice da NVIDIA para o JetPack (``onnxruntime-gpu``);
    o do PyPI só tem CPU — e então a inferência roda, mais devagar, e a telemetria diz.
    """

    def __init__(self) -> None:
        super().__init__(
            prefer=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            name="Jetson Orin NX 16 GB",
        )


_TEGRA_RAM = re.compile(r"RAM (\d+)/(\d+)MB")
_TEGRA_GPU = re.compile(r"GR3D_FREQ (\d+)%")
_TEGRA_TEMP = re.compile(r"\b(gpu|GPU)@(-?\d+(?:\.\d+)?)C")
_TEGRA_CPU = re.compile(r"CPU \[([^\]]*)\]")
_TEGRA_GPU_FREQ = re.compile(r"GR3D_FREQ \d+%@\[?(\d+)")
_TEGRA_TEMPS = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)@(-?\d+(?:\.\d+)?)C\b")


def parse_tegrastats(linha: str) -> dict[str, float | int | None]:
    """Uma linha do ``tegrastats`` → uso de GPU, memória e temperatura da GPU.

    Ex. (Orin): ``RAM 2448/15388MB (lfb 2x4MB) SWAP 0/7694MB CPU [3%@729,1%@729,...]
    GR3D_FREQ 0% cpu@44.25C ... gpu@43.875C ...``. Campo ausente → ``None``.
    """
    ram = _TEGRA_RAM.search(linha)
    gpu = _TEGRA_GPU.search(linha)
    temp = _TEGRA_TEMP.search(linha)
    cpu = _TEGRA_CPU.search(linha)
    cpu_pct = None
    if cpu:
        valores = [float(m) for m in re.findall(r"(\d+)%@", cpu.group(1))]
        cpu_pct = sum(valores) / len(valores) if valores else None
    freqs = [float(m) for m in re.findall(r"\d+%@(\d+)", cpu.group(1))] if cpu else []
    gpu_freq = _TEGRA_GPU_FREQ.search(linha)
    # Sensores que leem -256 °C estão desligados (ex.: ``PMIC@-256C``): fora.
    temperaturas = {nome: float(v) for nome, v in _TEGRA_TEMPS.findall(linha)
                    if float(v) > -100}
    return {
        "cpu_freq_mhz": sum(freqs) / len(freqs) if freqs else None,
        "gpu_freq_mhz": float(gpu_freq.group(1)) if gpu_freq else None,
        "temperaturas_c": temperaturas,
        "memory_used_bytes": int(ram.group(1)) * 1024 * 1024 if ram else None,
        "memory_total_bytes": int(ram.group(2)) * 1024 * 1024 if ram else None,
        "accelerator_percent": float(gpu.group(1)) if gpu else None,
        "accelerator_temperature_c": float(temp.group(2)) if temp else None,
        "cpu_percent": cpu_pct,
    }


class JetsonSystemMonitor:
    """CPU/memória pelo ``/proc`` e GPU/temperatura pelo ``tegrastats`` (Jetson).

    O ``tegrastats`` roda numa thread lendo uma linha por segundo; a leitura devolve a
    última. Sem ``tegrastats`` (fora da Jetson), a GPU sai ``None`` — não zero.
    """

    def __init__(self, comando: str = "tegrastats", intervalo_ms: int = 1000,
                 proc: Path | str = "/proc") -> None:
        self._linux = LinuxSystemMonitor(proc)
        self._ultima: dict[str, Any] = {}
        self._processo: subprocess.Popen | None = None
        if shutil.which(comando):
            self._processo = subprocess.Popen(
                [comando, "--interval", str(intervalo_ms)], stdout=subprocess.PIPE, text=True
            )
            threading.Thread(target=self._ler, daemon=True).start()

    def _ler(self) -> None:
        assert self._processo is not None and self._processo.stdout is not None
        for linha in self._processo.stdout:
            self._ultima = parse_tegrastats(linha)

    def ultima_tegrastats(self) -> dict[str, Any]:
        """A última linha do ``tegrastats`` já interpretada (clocks e todas as
        temperaturas) — o ensaio de calor lê daqui. Vazio fora da Jetson."""
        return dict(self._ultima)

    def read(self) -> SystemStats:
        base = self._linux.read()
        t = self._ultima
        return SystemStats(
            cpu_percent=base.cpu_percent,
            memory_used_bytes=base.memory_used_bytes,
            memory_total_bytes=base.memory_total_bytes,
            load_1m=base.load_1m,
            accelerator_percent=t.get("accelerator_percent"),
            accelerator_memory_used_bytes=None,  # memória unificada: já está em memória
            accelerator_temperature_c=t.get("accelerator_temperature_c"),
        )

    def describe(self) -> DeviceInfo:
        return DeviceInfo("sistema", "Jetson Orin NX 16 GB (tegrastats)", simulated=False,
                          detail={"tegrastats": self._processo is not None})


# ---------------------------------------------------------------------------------------------
# GNSS RTK: u-blox ZED-F9P (NMEA)
# ---------------------------------------------------------------------------------------------

_FIX_GGA = {0: "none", 1: "gps", 2: "gps", 4: "fixed", 5: "float", 6: "gps"}


def _nmea_ok(sentenca: str) -> bool:
    """Confere o checksum ``*hh`` (XOR dos bytes entre ``$`` e ``*``)."""
    if not sentenca.startswith("$") or "*" not in sentenca:
        return False
    corpo, _, soma = sentenca[1:].partition("*")
    calculado = 0
    for ch in corpo:
        calculado ^= ord(ch)
    try:
        return calculado == int(soma[:2], 16)
    except ValueError:
        return False


def _graus(valor: str, hemisferio: str) -> float | None:
    if not valor:
        return None
    ponto = valor.index(".")
    graus = float(valor[: ponto - 2]) + float(valor[ponto - 2:]) / 60.0
    return -graus if hemisferio in ("S", "W") else graus


@dataclass(slots=True)
class _EstadoGnss:
    lat: float | None = None
    lon: float | None = None
    alt_m: float | None = None
    fix: str = "none"
    hdop: float | None = None
    heading_deg: float | None = None
    speed_ms: float | None = None
    recorded_at: datetime | None = None


class NmeaParser:
    """GGA (posição, qualidade do fix, HDOP, altitude) e RMC (velocidade, rumo).

    Qualidade GGA → fix: 4 = ``fixed`` (RTK fixo), 5 = ``float`` (RTK flutuante),
    1/2/6 = ``gps``, 0 = ``none``. Sentença com checksum errado é descartada.
    """

    def __init__(self) -> None:
        self.estado = _EstadoGnss()

    def feed(self, sentenca: str, agora: datetime) -> bool:
        sentenca = sentenca.strip()
        if not _nmea_ok(sentenca):
            return False
        campos = sentenca[1:].split("*")[0].split(",")
        tipo = campos[0][-3:]
        if tipo == "GGA" and len(campos) >= 10:
            qualidade = int(campos[6] or 0)
            self.estado.fix = _FIX_GGA.get(qualidade, "gps")
            self.estado.lat = _graus(campos[2], campos[3])
            self.estado.lon = _graus(campos[4], campos[5])
            self.estado.hdop = float(campos[8]) if campos[8] else None
            self.estado.alt_m = float(campos[9]) if campos[9] else None
            self.estado.recorded_at = agora
            return True
        if tipo == "RMC" and len(campos) >= 9:
            if campos[7]:
                self.estado.speed_ms = float(campos[7]) * 0.514444  # nós → m/s
            if campos[8]:
                self.estado.heading_deg = float(campos[8])
            return True
        return False


class NmeaGnss:
    """ZED-F9P pela serial/USB (``/dev/ttyACM0``), lendo NMEA numa thread.

    As correções RTK (NTRIP ou rádio da base) chegam ao receptor por fora deste
    adapter (``str2str``/rádio): aqui só se lê o resultado. Fix mais velho que
    ``max_age_s`` → :class:`HardwareUnavailableError`.
    """

    def __init__(self, port: str, baudrate: int = 115200, clock: Clock | None = None,
                 max_age_s: float = 2.0, stream: io.TextIOBase | None = None) -> None:
        self.port = port
        self.clock = clock
        self.max_age_s = max_age_s
        self.parser = NmeaParser()
        self._lock = threading.Lock()
        if stream is None:
            import serial  # pyserial — extra [rover-v3]

            stream = io.TextIOWrapper(serial.Serial(port, baudrate, timeout=1),
                                      encoding="ascii", errors="ignore")
        self._stream = stream
        threading.Thread(target=self._ler, daemon=True).start()

    def _agora(self) -> datetime:
        return self.clock.now() if self.clock else datetime.now(UTC)

    def _ler(self) -> None:
        for linha in self._stream:
            with self._lock:
                self.parser.feed(linha, self._agora())

    def read(self) -> GpsFix:
        with self._lock:
            e = self.parser.estado
            if e.recorded_at is None or e.lat is None or e.lon is None:
                raise HardwareUnavailableError(f"GNSS {self.port}: sem posição ainda.")
            idade = (self._agora() - e.recorded_at).total_seconds()
            if idade > self.max_age_s:
                raise HardwareUnavailableError(
                    f"GNSS {self.port}: última posição há {idade:.0f} s."
                )
            return GpsFix(e.lat, e.lon, e.alt_m, e.fix, e.hdop, e.recorded_at,
                          e.heading_deg, e.speed_ms)

    def describe(self) -> DeviceInfo:
        return DeviceInfo("gps", "u-blox ZED-F9P (NMEA)", simulated=False,
                          detail={"port": self.port})


# ---------------------------------------------------------------------------------------------
# IMU: Xsens MTi-630 (CAN)
# ---------------------------------------------------------------------------------------------

XSENS_CAN_PADRAO = {
    "euler_id": 0x22,
    "euler_escala_graus": 2.0**-7,
    "acc_id": 0x34,
    "acc_escala_ms2": 2.0**-8,
}
"""IDs e escalas do *MT CAN protocol* (3 x int16 big-endian por mensagem). **Conferir** no MT
Manager do sensor instalado: os IDs são configuráveis no próprio MTi.

O giroscópio (*rate of turn*) só é lido se o ``hardware.json`` disser ``gyro_id`` e
``gyro_escala_dps`` (em °/s por unidade), conferidos no MT Manager — sem eles, o ensaio de
IMU deriva a taxa dos ângulos."""


def decode_xsens_int16x3(dados: bytes, escala: float) -> tuple[float, float, float]:
    if len(dados) < 6:
        raise ValueError(f"mensagem CAN com {len(dados)} bytes, esperava 6")
    a, b, c = np.frombuffer(bytes(dados[:6]), dtype=">i2").astype(float) * escala
    return float(a), float(b), float(c)


class XsensCanImu:
    """Xsens MTi-630 pelo barramento CAN (``python-can``, interface ``socketcan``).

    Uma thread lê as mensagens de ângulos de Euler e de aceleração e guarda as últimas.
    """

    def __init__(self, channel: str = "can0", bitrate: int = 500000, clock: Clock | None = None,
                 max_age_s: float = 1.0, bus: Any | None = None,
                 ids: dict[str, float] | None = None) -> None:
        self.channel = channel
        self.clock = clock
        self.max_age_s = max_age_s
        self.cfg = {**XSENS_CAN_PADRAO, **(ids or {})}
        self._euler: tuple[float, float, float] | None = None
        self._acc: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._gyro: tuple[float, float, float] | None = None
        self._quando: datetime | None = None
        if bus is None:
            import can  # python-can — extra [rover-v3]

            bus = can.interface.Bus(channel=channel, interface="socketcan", bitrate=bitrate)
        self._bus = bus
        threading.Thread(target=self._ler, daemon=True).start()

    def _agora(self) -> datetime:
        return self.clock.now() if self.clock else datetime.now(UTC)

    def handle(self, arbitration_id: int, data: bytes) -> None:
        if arbitration_id == int(self.cfg["euler_id"]):
            self._euler = decode_xsens_int16x3(data, float(self.cfg["euler_escala_graus"]))
            self._quando = self._agora()
        elif arbitration_id == int(self.cfg["acc_id"]):
            self._acc = decode_xsens_int16x3(data, float(self.cfg["acc_escala_ms2"]))
        elif "gyro_id" in self.cfg and arbitration_id == int(self.cfg["gyro_id"]):
            self._gyro = decode_xsens_int16x3(data, float(self.cfg["gyro_escala_dps"]))

    def _ler(self) -> None:
        while True:
            msg = self._bus.recv(timeout=1.0)
            if msg is not None:
                self.handle(msg.arbitration_id, bytes(msg.data))

    def read(self) -> ImuReading:
        if self._euler is None or self._quando is None:
            raise HardwareUnavailableError(f"IMU Xsens {self.channel}: sem leitura ainda.")
        idade = (self._agora() - self._quando).total_seconds()
        if idade > self.max_age_s:
            raise HardwareUnavailableError(f"IMU Xsens {self.channel}: leitura há {idade:.1f} s.")
        roll, pitch, yaw = self._euler
        return ImuReading(roll, pitch, yaw, self._acc, self._quando, self._gyro)

    def describe(self) -> DeviceInfo:
        return DeviceInfo("imu", "Xsens MTi-630 (CAN)", simulated=False,
                          detail={"channel": self.channel,
                                  "ids": {k: v for k, v in self.cfg.items() if k.endswith("_id")}})


# ---------------------------------------------------------------------------------------------
# Câmeras: Luxonis OAK-D Pro W PoE
# ---------------------------------------------------------------------------------------------


def depth_median_mm(depth: np.ndarray, roi: float = 0.5) -> float | None:
    """Mediana da profundidade (mm) no centro do quadro, ignorando zeros (sem medida).

    ``roi`` é a fração central (0,5 = metade do lado). A planta fica no centro do quadro
    lateral; as bordas pegam céu, chão e a fileira vizinha.
    """
    if depth.ndim != 2 or depth.size == 0:
        return None
    h, w = depth.shape
    dy, dx = int(h * (1 - roi) / 2), int(w * (1 - roi) / 2)
    centro = depth[dy:h - dy, dx:w - dx]
    validos = centro[centro > 0]
    if validos.size < max(16, centro.size // 20):  # menos de 5 % medido: não confia
        return None
    return float(np.median(validos))


class OakDCamera:
    """Uma OAK-D Pro W PoE: foto parada de 12 MP (JPEG no próprio sensor) + profundidade.

    Pipeline ``depthai`` 2.x: ``ColorCamera`` (12 MP) → ``VideoEncoder`` MJPEG para a foto
    parada; ``MonoCamera`` L/R → ``StereoDepth`` alinhada à RGB para a distância. A cada
    ``capture()`` pede uma foto parada e lê o último mapa de profundidade.

    **Não rodou numa OAK-D real** (ver o cabeçalho do módulo): o primeiro teste em bancada
    deve conferir a fila ``still`` e o alinhamento da profundidade.
    """

    def __init__(self, camera_id: str, ip: str, camera_height_mm: float,
                 clock: Clock | None = None, fps: float = 5.0, timeout_s: float = 3.0,
                 dai: Any | None = None, side: str | None = None,
                 resolution: str = "THE_12_MP", undistort: bool = True) -> None:
        if side not in (None, "left", "right"):
            raise ValueError(f"OAK-D {camera_id}: lado {side!r} (use left ou right)")
        self.side = side
        self.camera_id = camera_id
        self.ip = ip
        self.camera_height_mm = camera_height_mm
        self.clock = clock
        self.timeout_s = timeout_s
        self.resolution = resolution
        self._indice = 0
        self.undistorter = None
        self.lens_error: str | None = None
        if dai is None:
            import depthai as dai  # extra [rover-v3]
        self._dai = dai
        try:
            self._abrir(fps)
        except Exception as exc:  # o SDK levanta RuntimeError genérico
            raise HardwareUnavailableError(f"OAK-D {camera_id} ({ip}): {exc}") from exc
        if undistort:
            self._ler_lente()

    def _ler_lente(self) -> None:
        """A calibração de fábrica da OAK-D vira o mapa de correção (ver :mod:`.lente`).

        Sem calibração legível a câmera continua funcionando sem correção — e o diagnóstico
        mostra ``lente: sem calibração`` para o técnico decidir.
        """
        from valtexa_rover.hal.lente import LensModel, Undistorter

        try:
            calib = self._device.readCalibration()
            largura, altura = self._tamanho_foto()
            lente = LensModel.from_depthai(calib, self._dai, self._dai.CameraBoardSocket.CAM_A,
                                           largura, altura)
            self.undistorter = Undistorter(lente)
        except Exception as exc:  # calibração ausente, SDK diferente
            self.undistorter = None
            self.lens_error = f"{type(exc).__name__}: {exc}"

    def _tamanho_foto(self) -> tuple[int, int]:
        """Largura e altura da foto parada na resolução escolhida."""
        conhecidos = {"THE_12_MP": (4056, 3040), "THE_4_K": (3840, 2160),
                      "THE_1080_P": (1920, 1080), "THE_2024X1520": (2024, 1520),
                      "THE_5_MP": (2592, 1944), "THE_13_MP": (4208, 3120)}
        if self.resolution in conhecidos:
            return conhecidos[self.resolution]
        raise ValueError(f"resolução {self.resolution!r} sem tamanho conhecido")

    PREVIA_PX = (960, 720)
    """Saída de vídeo para o operador (4:3, como o sensor), separada da foto parada."""

    def _abrir(self, fps: float) -> None:
        dai = self._dai
        p = dai.Pipeline()
        cor = p.create(dai.node.ColorCamera)
        cor.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        # 12 MP por padrão; ``resolucao`` no hardware.json troca (ex.: THE_2024X1520, o
        # modo com binning, se o sensor instalado tiver — conferir no depthai da bancada).
        cor.setResolution(getattr(dai.ColorCameraProperties.SensorResolution, self.resolution))
        cor.setFps(fps)
        enc = p.create(dai.node.VideoEncoder)
        enc.setDefaultProfilePreset(1, dai.VideoEncoderProperties.Profile.MJPEG)
        cor.still.link(enc.input)
        saida_foto = p.create(dai.node.XLinkOut)
        saida_foto.setStreamName("still")
        enc.bitstream.link(saida_foto.input)
        # Vídeo ao vivo (estação de pilotagem): a prévia RGB pequena sai por outra fila e
        # nunca disputa a foto parada que a Edge AI usa.
        cor.setPreviewSize(*self.PREVIA_PX)
        cor.setInterleaved(True)
        cor.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        saida_previa = p.create(dai.node.XLinkOut)
        saida_previa.setStreamName("preview")
        cor.preview.link(saida_previa.input)
        controle = p.create(dai.node.XLinkIn)
        controle.setStreamName("control")
        controle.out.link(cor.inputControl)
        esq = p.create(dai.node.MonoCamera)
        esq.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        dir_ = p.create(dai.node.MonoCamera)
        dir_.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        estereo = p.create(dai.node.StereoDepth)
        estereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        estereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        esq.out.link(estereo.left)
        dir_.out.link(estereo.right)
        saida_prof = p.create(dai.node.XLinkOut)
        saida_prof.setStreamName("depth")
        estereo.depth.link(saida_prof.input)
        self._device = dai.Device(p, dai.DeviceInfo(self.ip))
        self._q_foto = self._device.getOutputQueue("still", maxSize=1, blocking=False)
        self._q_prof = self._device.getOutputQueue("depth", maxSize=1, blocking=False)
        self._q_ctrl = self._device.getInputQueue("control")
        try:
            self._q_previa = self._device.getOutputQueue("preview", maxSize=1, blocking=False)
        except Exception:  # firmware sem a prévia: o vídeo usa a foto parada reduzida
            self._q_previa = None

    def capture(self) -> Frame:
        dai = self._dai
        pedido = dai.CameraControl()
        pedido.setCaptureStill(True)
        try:
            self._q_ctrl.send(pedido)
            foto = self._q_foto.get()  # bloqueia até a foto parada chegar
            prof = self._q_prof.tryGet()
        except Exception as exc:
            raise HardwareUnavailableError(f"OAK-D {self.camera_id}: {exc}") from exc
        dados = bytes(foto.getData())
        hfov = None
        if self.undistorter is not None:
            try:
                dados, largura, altura = self.undistorter.apply_jpeg(dados)
                hfov = self.undistorter.hfov_deg
            except Exception as exc:  # foto que não decodifica: o Runtime decide
                raise HardwareUnavailableError(f"OAK-D {self.camera_id}: lente: {exc}") from exc
        else:
            from PIL import Image

            with Image.open(io.BytesIO(dados)) as img:
                largura, altura = img.size
        distancia = None if prof is None else depth_median_mm(np.asarray(prof.getFrame()))
        self._indice += 1
        quando = self.clock.now() if self.clock else datetime.now(UTC)
        return Frame(self.camera_id, self._indice, quando, dados, "image/jpeg", largura, altura,
                     source=f"oak-d:{self.ip}", distance_mm=distancia,
                     camera_height_mm=self.camera_height_mm, side=self.side, hfov_deg=hfov)

    def capture_preview(self, lado: int = 640, qualidade: int = 70
                        ) -> tuple[bytes, int, int, datetime]:
        """Um quadro do vídeo ao vivo (JPEG com lado maior ``lado``), pela prévia RGB.

        Sem correção de lente: é para o operador ver, não para a IA. Sem a fila da prévia,
        cai para a foto parada reduzida (mais lenta). **Não rodou numa OAK-D real.**
        """
        from PIL import Image

        if self._q_previa is None:
            from valtexa_rover.runtime.video import reduz

            f = self.capture()
            jpeg, w, h = reduz(f.data, lado, qualidade)
            return jpeg, w, h, f.captured_at
        try:
            pacote = self._q_previa.get()
            rgb = np.asarray(pacote.getFrame(), dtype=np.uint8)
        except Exception as exc:
            raise HardwareUnavailableError(f"OAK-D {self.camera_id}: prévia: {exc}") from exc
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise HardwareUnavailableError(
                f"OAK-D {self.camera_id}: prévia em formato {rgb.shape} (esperado HxWx3)")
        img = Image.fromarray(rgb)
        img.thumbnail((lado, lado))
        saida = io.BytesIO()
        img.save(saida, "JPEG", quality=qualidade)
        quando = self.clock.now() if self.clock else datetime.now(UTC)
        return saida.getvalue(), img.width, img.height, quando

    def describe(self) -> DeviceInfo:
        return DeviceInfo("camera", "Luxonis OAK-D Pro W PoE", simulated=False,
                          detail={"id": self.camera_id, "ip": self.ip,
                                  "altura_mm": self.camera_height_mm, "lado": self.side,
                                  "resolucao": self.resolution,
                                  "lente": (
                                      f"corrigida ({self.undistorter.lens.model}, "
                                      f"HFOV efetivo {self.undistorter.hfov_deg:.1f}°)"
                                      if self.undistorter is not None
                                      else f"sem correção ({self.lens_error or 'desligada'})")})


# ---------------------------------------------------------------------------------------------
# Navegação (fim de rota)
# ---------------------------------------------------------------------------------------------


class NavegacaoPorArquivo:
    """O Runtime e a pilha de navegação conversam por arquivos JSON em ``/run/valtexa``.

    * ``missao`` (Runtime → navegação): a missão em execução — ``id``, ``field_id`` e o
      ``payload`` assinado da plataforma, com a rota —, ou ``{"id": null}`` sem missão;
    * ``estado`` (navegação → Runtime): no mínimo ``{"mission_id": ..., "route_done": true}``
      ao terminar a rota. Sem o arquivo a rota está em curso; um ``route_done`` de outra
      missão é ignorado (``cli.rodar_servico``);
    * ``parada`` (Runtime → navegação): pedido de parada com motivo e ``at``. A navegação
      confirma no ``estado`` com ``parado_em`` (instante em que as rodas pararam) e
      ``parada_ref`` (o ``at`` que ela obedeceu); ``v_ms`` opcional. É o que o ensaio
      ``parada`` mede.

    O operador pode encerrar a rota à mão:
    ``echo '{"route_done": true}' > /run/valtexa/navegacao.json``. A parada de emergência
    continua no microcontrolador de segurança, não aqui.
    """

    def __init__(self, estado: Path | str, parada: Path | str, missao: Path | str | None = None,
                 *, clock: Clock | None = None, validade_s: float | None = None) -> None:
        self.estado = Path(estado)
        self.parada = Path(parada)
        self.missao = Path(missao) if missao else self.estado.with_name("missao.json")
        self.clock = clock
        self.validade_s = validade_s

    def _agora(self) -> datetime:
        return self.clock.now() if self.clock else datetime.now(UTC)

    def state(self) -> dict:
        try:
            bruto = json.loads(self.estado.read_text(encoding="utf-8"))
            idade = self._agora().timestamp() - self.estado.stat().st_mtime
        except FileNotFoundError:
            return {"route_done": False, "navegacao": "sem estado"}
        except (OSError, ValueError) as exc:  # escrita pela metade: tenta no próximo ciclo
            return {"route_done": False, "navegacao": f"estado ilegível: {exc}"}
        if not isinstance(bruto, dict):
            return {"route_done": False, "navegacao": "estado não é um objeto JSON"}
        if self.validade_s is not None and idade > self.validade_s:
            # Navegação parada de publicar: não conclui missão por um estado antigo.
            return {**bruto, "route_done": False, "navegacao": f"estado velho ({idade:.0f} s)"}
        return {**bruto, "route_done": bool(bruto.get("route_done", False))}

    def _escrever(self, destino: Path, dados: dict) -> None:
        destino.parent.mkdir(parents=True, exist_ok=True)
        temporario = destino.with_suffix(".tmp")
        temporario.write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")
        os.replace(temporario, destino)  # a navegação nunca lê o arquivo pela metade

    def announce_mission(self, missao: Any) -> None:
        if missao is None:
            self._escrever(self.missao, {"id": None, "at": self._agora().isoformat()})
            return
        self._escrever(self.missao, {"id": missao.id, "field_id": missao.field_id,
                                     "payload": missao.payload,
                                     "at": self._agora().isoformat()})

    def stop(self, reason: str) -> dict:
        """Escreve o pedido e o devolve: a navegação confirma copiando ``at`` para o
        ``parada_ref`` do ``navegacao.json``, com ``parado_em`` (ver ``ensaios.seguranca``)."""
        pedido = {"reason": reason, "at": self._agora().isoformat()}
        self._escrever(self.parada, pedido)
        return pedido

    @property
    def controle(self) -> Path:
        """``controle.json``: o modo que o centro de comando pediu."""
        return self.estado.with_name("controle.json")

    @property
    def teleop_path(self) -> Path:
        """``teleop.json``: o ponto do controle remoto (vence em ``valid_until``)."""
        return self.estado.with_name("teleop.json")

    def set_mode(self, modo: str) -> None:
        """``autonomo`` (segue a rota da missão), ``manual`` (só o controle remoto),
        ``pausado`` (para e espera) ou ``parado``. A navegação lê ``controle.json``."""
        if modo not in ("autonomo", "manual", "pausado", "parado"):
            raise ValueError(f"modo {modo!r}")
        self._escrever(self.controle, {"modo": modo, "at": self._agora().isoformat()})
        if modo == "parado":
            self.stop("modo parado pelo centro de comando")

    def mode(self) -> str | None:
        try:
            return json.loads(self.controle.read_text(encoding="utf-8")).get("modo")
        except (OSError, ValueError):
            return None

    def teleop(self, v: float, w: float, valid_until: str | None, seq: int = 0) -> None:
        """Ponto de ajuste para a navegação/motores. Vencido ``valid_until``, ela para
        (homem-morto) — o Runtime também escreve zero quando o ponto some."""
        self._escrever(self.teleop_path, {"seq": seq, "v": float(v), "w": float(w),
                                          "valid_until": valid_until or self._agora().isoformat()})

    def describe(self) -> DeviceInfo:
        return DeviceInfo("controller", "navegação por arquivo", simulated=False,
                          detail={"estado": str(self.estado), "parada": str(self.parada),
                                  "missao": str(self.missao)})


# ---------------------------------------------------------------------------------------------
# Montagem pelo hardware.json
# ---------------------------------------------------------------------------------------------


def load_hardware_config(dados: Path) -> dict[str, Any]:
    caminho = Path(os.environ.get("VALTEXA_HARDWARE") or Path(dados) / "hardware.json")
    if not caminho.exists():
        raise HardwareUnavailableError(
            f"{caminho} não existe: copie deploy/hardware.rover-v3.json.exemplo e ajuste."
        )
    return json.loads(caminho.read_text(encoding="utf-8"))


def hal_rover_v3(*, dados: Path, url: str, clock: Clock, **fabricas: Any) -> Hardware:
    """O Hardware do Rover V3 a partir do ``hardware.json`` (ver o exemplo em ``deploy/``).

    ``fabricas`` troca a construção de uma peça (testes): ``camera``, ``gnss``, ``imu``.
    Sem ``navegacao`` no arquivo não há controlador e a missão não termina sozinha.
    """
    cfg = load_hardware_config(dados)
    fazer_camera = fabricas.get("camera") or (
        lambda c: OakDCamera(c["id"], c["ip"], float(c["altura_mm"]), clock=clock,
                             fps=float(c.get("fps", 5.0)), side=c.get("lado"),
                             resolution=c.get("resolucao", "THE_12_MP"),
                             undistort=bool(c.get("corrigir_lente", True))))
    fazer_gnss = fabricas.get("gnss") or (
        lambda g: NmeaGnss(g["porta"], int(g.get("baud", 115200)), clock=clock))
    fazer_imu = fabricas.get("imu") or (
        lambda i: XsensCanImu(i.get("canal", "can0"), int(i.get("bitrate", 500000)), clock=clock,
                              ids=i.get("ids")))
    cameras = [fazer_camera(c) for c in cfg["cameras"]]
    if not cameras:
        raise HardwareUnavailableError("hardware.json sem câmeras.")
    imu = fazer_imu(cfg["imu"]) if cfg.get("imu") else None
    nav = cfg.get("navegacao")
    controle = None if not nav else NavegacaoPorArquivo(
        nav.get("estado", "/run/valtexa/navegacao.json"),
        nav.get("parada", "/run/valtexa/parar.json"), nav.get("missao"), clock=clock,
        validade_s=nav.get("validade_s"))
    cota = cfg.get("armazenamento", {}).get("cota_bytes")
    rede_cfg = cfg.get("rede", {})
    if rede_cfg.get("modo") == "base":  # rádio da fazenda: a nuvem está do outro lado da base
        from valtexa_rover.hal.rede_base import rede_via_base

        rede = rede_via_base(rede_cfg, Path(dados))
    else:
        rede = HttpNetwork(url, timeout_s=float(rede_cfg.get("timeout_s", 30.0)))
    return Hardware(
        cameras=cameras,
        gps=fazer_gnss(cfg["gnss"]),
        imu=imu,
        # Bateria e ambiente vêm do microcontrolador de segurança (protocolo a definir).
        sensors=None,
        storage=LocalDiskStorage(dados, quota_bytes=cota),
        network=rede,
        accelerator=JetsonAccelerator(),
        # Movimento é do microcontrolador de segurança; daqui só o fim de rota e a parada.
        controller=controle,
        hardware_version=cfg.get("versao", HARDWARE_VERSION),
        system=JetsonSystemMonitor() if cfg.get("tegrastats", True) else LinuxSystemMonitor(),
    )
