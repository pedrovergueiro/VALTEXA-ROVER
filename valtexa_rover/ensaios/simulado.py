"""HAL de bancada **SIMULADA** para os ensaios, com injeção de falha.

Brief, seção 55: simulação marcada como simulação. Tudo aqui diz ``simulated=True`` e
serve para testar os ensaios (e rodar a FASE 0 no simulador) sem hardware. Nenhum número
que sai daqui diz como o Rover V3 de verdade se comporta.

Cada peça aceita falhas: câmera que perde quadro, que pula índice, que atrasa o carimbo;
profundidade com viés; GNSS que demora a fixar, com ruído, viés e perdas; IMU com eixos
trocados, escala errada e giroscópio ruidoso; navegação que demora ou não confirma a
parada; rede lenta ou que perde; Jetson que esquenta e baixa o clock.
"""

from __future__ import annotations

import io
import math
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from valtexa_rover.clock import Clock
from valtexa_rover.ensaios.estatistica import GRAVIDADE_PADRAO, RAIO_TERRA_M
from valtexa_rover.ensaios.relogio import FonteDoRelogio
from valtexa_rover.hal import relogio as chrony
from valtexa_rover.hal.interfaces import (
    DeviceInfo,
    Frame,
    GpsFix,
    Hardware,
    HardwareUnavailableError,
    ImuReading,
    NetworkUnavailableError,
)
from valtexa_rover.hal.local import LocalDiskStorage, OnnxCpuAccelerator
from valtexa_rover.hal.rover_v3 import NavegacaoPorArquivo
from valtexa_rover.hal.simulated import SimulatedSensors, SimulatedSystemMonitor

_JPEG_CACHE: dict[tuple[int, int], bytes] = {}


class RelogioDeBancada:
    """SIMULAÇÃO: horário de verdade, esperas instantâneas.

    ``now()`` é o relógio do sistema (as requisições e os comandos assinados conferem o
    horário com a plataforma); ``sleep`` não dorme — só adianta ``monotonic``. Assim a
    FASE 0 inteira (2 h de calor, minutos de GNSS) roda em segundos contra uma plataforma
    de verdade, e o que é medido de verdade (RTT da rede) continua medido.
    """

    def __init__(self) -> None:
        self._extra = 0.0
        self._lock = threading.Lock()

    def now(self):
        from datetime import datetime

        from valtexa_rover._compat import UTC

        return datetime.now(UTC)

    def monotonic(self) -> float:
        import time

        with self._lock:
            return time.monotonic() + self._extra

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self._extra += max(0.0, seconds)


def _jpeg(largura: int = 64, altura: int = 48) -> bytes:
    if (largura, altura) not in _JPEG_CACHE:
        saida = io.BytesIO()
        Image.new("RGB", (largura, altura), (40, 140, 50)).save(saida, "JPEG", quality=80)
        _JPEG_CACHE[(largura, altura)] = saida.getvalue()
    return _JPEG_CACHE[(largura, altura)]


class CameraDeBancada:
    """SIMULAÇÃO: OAK-D de mentira, com foto e mapa de profundidade.

    Falhas: ``falha_em`` (índices de captura que levantam), ``salto_em`` (capturas em que
    o ``frame_index`` pula um), ``atraso_carimbo_s`` (carimbo fora do instante real),
    ``vies_mm``/``ruido_mm``/``preenchimento`` na profundidade.
    """

    def __init__(self, camera_id: str, clock: Clock, *, distancia_mm: float = 500.0,
                 vies_mm: float = 0.0, ruido_mm: float = 3.0, preenchimento: float = 0.95,
                 falha_em: Iterable[int] = (), salto_em: Iterable[int] = (),
                 atraso_carimbo_s: float = 0.0, semente: int = 0) -> None:
        self.camera_id = camera_id
        self.clock = clock
        self.distancia_mm = distancia_mm
        self.vies_mm = vies_mm
        self.ruido_mm = ruido_mm
        self.preenchimento = preenchimento
        self.falha_em = set(falha_em)
        self.salto_em = set(salto_em)
        self.atraso_carimbo_s = atraso_carimbo_s
        self._rng = np.random.default_rng(semente)
        self._capturas = 0
        self._indice = 0
        self._lock = threading.Lock()

    def capture(self) -> Frame:
        with self._lock:
            n = self._capturas
            self._capturas += 1
            if n in self.falha_em:
                raise HardwareUnavailableError(f"{self.camera_id}: falha simulada na captura {n}")
            self._indice += 2 if n in self.salto_em else 1
            indice = self._indice
        quando = self.clock.now() + timedelta(seconds=self.atraso_carimbo_s)
        return Frame(self.camera_id, indice, quando, _jpeg(), "image/jpeg", 64, 48,
                     source="bancada simulada", distance_mm=self.distancia_mm,
                     camera_height_mm=1000.0, side="left", hfov_deg=95.0)

    def capture_depth(self) -> np.ndarray:
        with self._lock:
            mapa = (self.distancia_mm + self.vies_mm
                    + self._rng.normal(0, self.ruido_mm, (60, 80)))
            buracos = self._rng.random((60, 80)) > self.preenchimento
        mapa[buracos] = 0
        return np.clip(mapa, 0, 65535).astype(np.uint16)

    def describe(self) -> DeviceInfo:
        return DeviceInfo("camera", "OAK-D de bancada", True, {"id": self.camera_id})


class GnssRtkDeBancada:
    """SIMULAÇÃO: ZED-F9P parado sobre um ponto. ``tempo_fix_s`` até ``fixed``
    (antes, ``gps`` e depois ``float``), ruído gaussiano (cm), viés (leste, norte em cm) e
    ``perdas``: janelas ``(início, fim)`` em segundos desde a criação, em ``float``."""

    def __init__(self, clock: Clock, lat: float = -21.25, lon: float = -45.0, *,
                 tempo_fix_s: float = 20.0, ruido_cm: float = 0.8,
                 vies_cm: tuple[float, float] = (0.0, 0.0),
                 perdas: Iterable[tuple[float, float]] = (), semente: int = 0) -> None:
        self.clock = clock
        self.lat, self.lon = lat, lon
        self.tempo_fix_s = tempo_fix_s
        self.ruido_cm = ruido_cm
        self.vies_cm = vies_cm
        self.perdas = list(perdas)
        self._rng = np.random.default_rng(semente)
        self._inicio = clock.monotonic()

    def read(self) -> GpsFix:
        t = self.clock.monotonic() - self._inicio
        if t < 1.0:
            raise HardwareUnavailableError("GNSS sem posição ainda (simulação)")
        if any(a <= t < b for a, b in self.perdas) or t < self.tempo_fix_s:
            fix = "gps" if t < self.tempo_fix_s / 2 else "float"
            ruido = self.ruido_cm * 40
        else:
            fix, ruido = "fixed", self.ruido_cm
        e = (self.vies_cm[0] + self._rng.normal(0, ruido)) / 100
        n = (self.vies_cm[1] + self._rng.normal(0, ruido)) / 100
        lat = self.lat + math.degrees(n / RAIO_TERRA_M)
        lon = self.lon + math.degrees(e / (RAIO_TERRA_M * math.cos(math.radians(self.lat))))
        return GpsFix(lat, lon, 880.0, fix, 0.6, self.clock.now(), None, 0.0)

    def describe(self) -> DeviceInfo:
        return DeviceInfo("gps", "ZED-F9P de bancada", True)


class ImuDeBancada:
    """SIMULAÇÃO: MTi parado em ``roll``/``pitch``. Falhas: ``escala_g`` (acelerômetro
    descalibrado), ``eixos_trocados`` (x↔y, MTi montado girado), giroscópio com
    ``ruido_gyro_dps``/``vies_gyro_dps``; ``com_gyro=False`` imita o CAN sem *rate of
    turn*."""

    def __init__(self, clock: Clock, roll: float = 0.0, pitch: float = 0.0, *,
                 escala_g: float = 1.0, eixos_trocados: bool = False,
                 ruido_acc_ms2: float = 0.01, ruido_angulo_deg: float = 0.02,
                 ruido_gyro_dps: float = 0.03, vies_gyro_dps: float = 0.005,
                 com_gyro: bool = True, semente: int = 0) -> None:
        self.clock = clock
        self.roll, self.pitch = roll, pitch
        self.escala_g = escala_g
        self.eixos_trocados = eixos_trocados
        self.ruido_acc = ruido_acc_ms2
        self.ruido_angulo = ruido_angulo_deg
        self.ruido_gyro = ruido_gyro_dps
        self.vies_gyro = vies_gyro_dps
        self.com_gyro = com_gyro
        self._rng = np.random.default_rng(semente)

    def read(self) -> ImuReading:
        g = GRAVIDADE_PADRAO * self.escala_g
        r, p = math.radians(self.roll), math.radians(self.pitch)
        ax, ay, az = -g * math.sin(p), g * math.cos(p) * math.sin(r), g * math.cos(p) * math.cos(r)
        if self.eixos_trocados:
            ax, ay = ay, ax
        ruido = self._rng.normal(0, self.ruido_acc, 3)
        gyro = None
        if self.com_gyro:
            gyro = tuple(float(v) for v in self.vies_gyro + self._rng.normal(0, self.ruido_gyro,
                                                                             3))
        ang = self._rng.normal(0, self.ruido_angulo, 3)
        return ImuReading(self.roll + ang[0], self.pitch + ang[1], 90.0 + ang[2],
                          (ax + ruido[0], ay + ruido[1], az + ruido[2]), self.clock.now(), gyro)

    def describe(self) -> DeviceInfo:
        return DeviceInfo("imu", "MTi de bancada", True)


class NavegacaoDeBancada(NavegacaoPorArquivo):
    """SIMULAÇÃO: a pilha de navegação do outro lado dos arquivos. Confirma cada pedido de
    parada ``latencia_s`` depois (``parado_em``/``parada_ref`` no ``navegacao.json``), ou
    nunca (``confirma=False``); ``latencias`` dá uma por pedido, em ordem."""

    def __init__(self, pasta: Path, clock: Clock, *, latencia_s: float = 0.12,
                 latencias: list[float] | None = None, confirma: bool = True) -> None:
        pasta.mkdir(parents=True, exist_ok=True)
        super().__init__(pasta / "navegacao.json", pasta / "parar.json", clock=clock)
        self.latencia_s = latencia_s
        self.latencias = list(latencias or [])
        self.confirma = confirma
        self._pedido: tuple[str, float, float] | None = None  # (at, mono, latência)

    def stop(self, reason: str) -> dict:
        pedido = super().stop(reason)
        latencia = self.latencias.pop(0) if self.latencias else self.latencia_s
        self._pedido = (pedido["at"], self.clock.monotonic(), latencia)
        return pedido

    def state(self) -> dict:
        if self._pedido is not None and self.confirma:
            at, mono, latencia = self._pedido
            if self.clock.monotonic() - mono >= latencia:
                parado = self.clock.now() - timedelta(
                    seconds=self.clock.monotonic() - mono - latencia)
                self._escrever(self.estado, {"route_done": False, "v_ms": 0.0,
                                             "parado_em": parado.isoformat(),
                                             "parada_ref": at})
                self._pedido = None
        return super().state()

    def describe(self) -> DeviceInfo:
        return DeviceInfo("controller", "navegação de bancada", True)


class TermicaDeBancada:
    """SIMULAÇÃO: Jetson que esquenta ``aquece_c_por_min`` até ``temp_teto_c``; a partir
    de ``throttle_c`` os clocks caem ``queda_clock`` (fração) e a vazão junto."""

    def __init__(self, clock: Clock, *, temp_inicial_c: float = 45.0,
                 aquece_c_por_min: float = 1.0, temp_teto_c: float = 70.0,
                 throttle_c: float | None = None, queda_clock: float = 0.3) -> None:
        self.clock = clock
        self.temp_inicial = temp_inicial_c
        self.aquece = aquece_c_por_min
        self.teto = temp_teto_c
        self.throttle_c = throttle_c
        self.queda = queda_clock
        self._inicio = clock.monotonic()

    def temperatura(self) -> float:
        minutos = (self.clock.monotonic() - self._inicio) / 60
        return min(self.teto, self.temp_inicial + self.aquece * minutos)

    def fator(self) -> float:
        if self.throttle_c is not None and self.temperatura() >= self.throttle_c:
            return 1.0 - self.queda
        return 1.0

    def ler(self) -> dict[str, Any]:
        f = self.fator()
        return {"temp_c": round(self.temperatura(), 2), "cpu_freq_mhz": 1984.0 * f,
                "gpu_freq_mhz": 918.0 * f, "gpu_pct": 95.0}


@dataclass
class CargaDeBancada:
    """SIMULAÇÃO: uma "inferência" leva ``1 / (qps · fator térmico)`` segundos no relógio."""

    clock: Clock
    termica: TermicaDeBancada | None = None
    qps: float = 8.0

    def __call__(self) -> int:
        fator = self.termica.fator() if self.termica is not None else 1.0
        self.clock.sleep(1.0 / (self.qps * fator))
        return 1


@dataclass
class RedeDeBancada:
    """SIMULAÇÃO: enlace com ``rtt_s``, ``kbps`` e perda nos pings ``perde_em``."""

    clock: Clock
    rtt_s: float = 0.08
    kbps: float = 4000.0
    perde_em: set[int] = field(default_factory=set)
    _n: int = 0

    def ping(self) -> bool:
        n, self._n = self._n, self._n + 1
        if n in self.perde_em:
            self.clock.sleep(self.rtt_s * 4)
            raise NetworkUnavailableError("perda simulada")
        self.clock.sleep(self.rtt_s)
        return True

    def upload(self, n: int) -> int:
        self.clock.sleep(self.rtt_s + 8 * n / (self.kbps * 1000))
        return n

    def download(self, n: int) -> int:
        return self.upload(n)


class _RedeDesligada:
    def is_online(self) -> bool:
        return False

    def send(self, *a, **k):
        raise NetworkUnavailableError("bancada sem rede (simulação)")

    def describe(self) -> DeviceInfo:
        return DeviceInfo("rede", "sem rede", True)


def chrony_de_bancada(offset_s: float = -0.0000025, fonte: str = "PPS"):
    """SIMULAÇÃO: o que o ``chronyc -c`` diria com o GNSS/PPS travado (formato do chrony 4.x).

    ``offset_s`` e ``fonte`` (``"PPS"``, ``"GPS"``, um servidor NTP...) injetam defeito."""
    tracking = (f"50505300,{fonte},1,1791000000.123,{offset_s:.9f},0.000000400,0.000001100,"
                "-12.345,0.001,0.010,0.000000001,0.000010000,16.0,Normal\n")
    marca = "#" if fonte.isupper() else "^"
    sources = f"{marca},*,{fonte},0,4,377,10,0.000000010,0.000000012,0.000000100\n"
    return lambda: chrony.le_chrony(lambda a: tracking if a[-1] == "tracking" else sources)


def bancada(pasta: Path, clock: Clock, *, cameras: int = 8, falhas: dict | None = None
            ) -> tuple[Hardware, dict[str, Any]]:
    """O Rover V3 de bancada inteiro, simulado: ``(hardware, fontes)``.

    ``falhas`` injeta defeitos por peça, ex.: ``{"camera": {"falha_em": [3]},
    "gnss": {"tempo_fix_s": 500}, "imu": {"eixos_trocados": True},
    "navegacao": {"confirma": False}, "termica": {"throttle_c": 50},
    "rede": {"perde_em": {0, 1}}, "relogio": {"offset_s": 0.5}}``. ``camera`` vale para
    todas; ``camera:<n>`` para uma.
    """
    falhas = falhas or {}
    pasta.mkdir(parents=True, exist_ok=True)
    lista = []
    for i in range(cameras):
        opcoes = {**falhas.get("camera", {}), **falhas.get(f"camera:{i}", {})}
        lista.append(CameraDeBancada(f"oak-{i}", clock, semente=i, **opcoes))
    termica = TermicaDeBancada(clock, **falhas.get("termica", {}))
    hardware = Hardware(
        cameras=lista,
        gps=GnssRtkDeBancada(clock, **falhas.get("gnss", {})),
        imu=ImuDeBancada(clock, **falhas.get("imu", {})),
        sensors=SimulatedSensors(clock),
        storage=LocalDiskStorage(pasta / "dados"),
        network=_RedeDesligada(),
        accelerator=OnnxCpuAccelerator(),
        controller=NavegacaoDeBancada(pasta / "run", clock, **falhas.get("navegacao", {})),
        hardware_version="Rover V3 de bancada (simulado)",
        system=SimulatedSystemMonitor(),
    )
    fontes = {"termica": termica, "carga": CargaDeBancada(clock, termica),
              "rede": RedeDeBancada(clock, **falhas.get("rede", {})),
              "relogio": FonteDoRelogio(le_chrony=chrony_de_bancada(**falhas.get("relogio", {})))}
    return hardware, fontes


def _bancada_cli(*, dados: Path, url: str, clock: Clock) -> Hardware:
    """SIMULAÇÃO: a bancada como fábrica de HAL do ``valtexa-rover`` (testes do CLI)."""
    return bancada(Path(dados), clock)[0]
