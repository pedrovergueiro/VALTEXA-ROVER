"""Adapters **SIMULADOS** da HAL — para o simulador, os testes e o desenvolvimento.

Brief, seção 55: simulação marcada como simulação. Todo adapter daqui devolve
``simulated=True`` em ``describe()``, e o Runtime grava ``hardware.simulated``
em cada resultado da Edge AI. Nenhum deles conversa com hardware.

Os adapters **reais** que não dependem de hardware específico (disco local,
rede HTTP, CPU pelo ONNX Runtime) ficam em :mod:`valtexa_rover.hal.local`.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import math
from collections.abc import Callable, Sequence
from pathlib import Path

from PIL import Image

from valtexa_rover.clock import Clock
from valtexa_rover.hal.interfaces import (
    DeviceInfo,
    Frame,
    GpsFix,
    HardwareUnavailableError,
    HttpResponse,
    ImuReading,
    NetworkUnavailableError,
    SensorReading,
    SystemStats,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
CONTENT_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                 ".webp": "image/webp"}


class FolderCamera:
    """SIMULAÇÃO: "câmera" que lê, em ordem, as imagens de uma pasta.

    ``loop=True`` recomeça do início quando acaba (missões longas com poucas
    imagens); ``loop=False`` levanta :class:`HardwareUnavailableError`, como uma
    câmera que parou. ``fail_on`` lista índices de captura que falham — para
    testar câmera indisponível no meio da missão.
    """

    def __init__(
        self,
        camera_id: str,
        folder: Path | str,
        clock: Clock,
        *,
        loop: bool = True,
        fail_on: Sequence[int] = (),
    ) -> None:
        self.camera_id = camera_id
        self.folder = Path(folder)
        self.clock = clock
        self.loop = loop
        self.fail_on = set(fail_on)
        self.files = sorted(
            p for p in self.folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.files:
            raise HardwareUnavailableError(f"{camera_id}: nenhuma imagem em {self.folder}")
        self._index = 0

    def capture(self) -> Frame:
        index = self._index
        self._index += 1
        if index in self.fail_on:
            raise HardwareUnavailableError(f"{self.camera_id}: falha simulada na captura {index}")
        if index >= len(self.files) and not self.loop:
            raise HardwareUnavailableError(f"{self.camera_id}: sem mais imagens (simulação)")
        path = self.files[index % len(self.files)]
        data = path.read_bytes()
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
        return Frame(
            camera_id=self.camera_id,
            frame_index=index,
            captured_at=self.clock.now(),
            data=data,
            content_type=CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream"),
            width=width,
            height=height,
            source=path.name,
        )

    def describe(self) -> DeviceInfo:
        return DeviceInfo(
            "camera", "pasta de imagens", True,
            {"camera_id": self.camera_id, "imagens": len(self.files)},
        )


class TrajectoryGps:
    """SIMULAÇÃO: GPS que percorre uma trajetória ``[(lon, lat), ...]`` a velocidade fixa.

    A posição sai do tempo decorrido no relógio injetado; acabou a trajetória,
    fica parado no último ponto. ``lost_between`` simula perda de sinal entre dois
    instantes (segundos desde o início).
    """

    def __init__(
        self,
        waypoints: Sequence[tuple[float, float]],
        clock: Clock,
        *,
        speed_ms: float = 1.0,
        alt_m: float = 1.2,
        fix: str = "fixed",
        hdop: float = 0.7,
        lost_between: tuple[float, float] | None = None,
    ) -> None:
        if len(waypoints) < 1:
            raise ValueError("Trajetória vazia.")
        self.waypoints = list(waypoints)
        self.clock = clock
        self.speed_ms = speed_ms
        self.alt_m = alt_m
        self.fix = fix
        self.hdop = hdop
        self.lost_between = lost_between
        self._start = clock.monotonic()

    def read(self) -> GpsFix:
        elapsed = self.clock.monotonic() - self._start
        if self.lost_between and self.lost_between[0] <= elapsed < self.lost_between[1]:
            raise HardwareUnavailableError("GPS sem sinal (simulação)")
        lon, lat, heading = _along(self.waypoints, elapsed * self.speed_ms)
        return GpsFix(
            lat=lat, lon=lon, alt_m=self.alt_m, fix=self.fix, hdop=self.hdop,
            recorded_at=self.clock.now(), heading_deg=heading, speed_ms=self.speed_ms,
        )

    def describe(self) -> DeviceInfo:
        return DeviceInfo("gps", "trajetória", True, {"pontos": len(self.waypoints)})


class SimulatedImu:
    """SIMULAÇÃO: IMU de robô nivelado, com o rumo do GPS quando houver."""

    def __init__(self, clock: Clock, gps: TrajectoryGps | None = None) -> None:
        self.clock = clock
        self.gps = gps

    def read(self) -> ImuReading:
        yaw = 0.0
        if self.gps is not None:
            with contextlib.suppress(HardwareUnavailableError):
                yaw = self.gps.read().heading_deg or 0.0
        return ImuReading(0.0, 0.0, yaw, (0.0, 0.0, 9.81), self.clock.now())

    def describe(self) -> DeviceInfo:
        return DeviceInfo("imu", "nivelado", True)


class SimulatedSensors:
    """SIMULAÇÃO: temperatura interna que sobe devagar e bateria que desce."""

    def __init__(
        self,
        clock: Clock,
        *,
        temperature_c: float = 30.0,
        warming_c_per_h: float = 2.0,
        battery_percent: float = 100.0,
        drain_percent_per_h: float = 8.0,
    ) -> None:
        self.clock = clock
        self.base_temperature = temperature_c
        self.warming = warming_c_per_h
        self.base_battery = battery_percent
        self.drain = drain_percent_per_h
        self._start = clock.monotonic()

    def read(self) -> SensorReading:
        horas = (self.clock.monotonic() - self._start) / 3600
        return SensorReading(
            {
                "temperature_c": round(self.base_temperature + self.warming * horas, 2),
                "battery_percent": round(max(0.0, self.base_battery - self.drain * horas), 2),
            },
            self.clock.now(),
        )

    def describe(self) -> DeviceInfo:
        return DeviceInfo("sensores", "temperatura e bateria", True)


class SimulatedSystemMonitor:
    """SIMULAÇÃO: CPU, memória e acelerador de um computador embarcado imaginário.

    Os valores são fixos (ou os que o teste pedir): servem para exercitar a
    telemetria, a frota e os alertas sem hardware — nunca para dizer como um robô
    real se comporta.
    """

    def __init__(self, *, cpu_percent: float = 35.0, memory_used_bytes: int = 3 * 1024**3,
                 memory_total_bytes: int = 8 * 1024**3, accelerator_percent: float = 40.0,
                 accelerator_temperature_c: float = 55.0) -> None:
        self.cpu_percent = cpu_percent
        self.memory_used_bytes = memory_used_bytes
        self.memory_total_bytes = memory_total_bytes
        self.accelerator_percent = accelerator_percent
        self.accelerator_temperature_c = accelerator_temperature_c

    def read(self) -> SystemStats:
        return SystemStats(
            cpu_percent=self.cpu_percent, memory_used_bytes=self.memory_used_bytes,
            memory_total_bytes=self.memory_total_bytes, load_1m=None,
            accelerator_percent=self.accelerator_percent,
            accelerator_temperature_c=self.accelerator_temperature_c,
        )

    def describe(self) -> DeviceInfo:
        return DeviceInfo("sistema", "computador embarcado simulado", True)


Handler = Callable[[str, str, bytes, dict[str, str]], HttpResponse]


class FlakyNetwork:
    """SIMULAÇÃO: rede que cai e volta, na frente de um ``handler`` que responde.

    O ``handler`` é quem faz o papel da nuvem: nos testes do pacote, um servidor
    em memória; no teste de integração do SaaS, o ``TestClient`` da API real.

    * ``set_online(False)`` derruba o enlace até alguém religar;
    * ``drop_after(n)``: deixa passar mais ``n`` requisições e cai — é como se
      simula a queda no meio de um upload (ex.: em 70 % dos chunks);
    * ``sent`` conta as requisições que chegaram ao outro lado.
    """

    def __init__(self, handler: Handler, *, online: bool = True) -> None:
        self.handler = handler
        self._online = online
        self._budget: int | None = None
        self.sent = 0
        self.refused = 0

    def set_online(self, online: bool) -> None:
        self._online = online
        if online:
            self._budget = None

    def drop_after(self, requests: int) -> None:
        self._budget = requests

    def is_online(self) -> bool:
        return self._online

    def send(self, method: str, path: str, body: bytes, headers: dict[str, str]) -> HttpResponse:
        if self._online and self._budget is not None:
            if self._budget <= 0:
                self._online = False
            else:
                self._budget -= 1
        if not self._online:
            self.refused += 1
            raise NetworkUnavailableError("sem enlace (simulação)")
        self.sent += 1
        return self.handler(method, path, body, headers)

    def describe(self) -> DeviceInfo:
        return DeviceInfo("rede", "enlace intermitente", True, {"online": self._online})


class SimulatedController:
    """SIMULAÇÃO: controlador que só registra os pedidos de parada."""

    def __init__(self) -> None:
        self.stops: list[str] = []

    def state(self) -> dict:
        return {"moving": not self.stops, "stops": len(self.stops)}

    def stop(self, reason: str) -> None:
        self.stops.append(reason)

    def describe(self) -> DeviceInfo:
        return DeviceInfo("controlador", "registro de comandos", True)


def _along(points: list[tuple[float, float]], distance_m: float) -> tuple[float, float, float]:
    """Ponto a ``distance_m`` metros do início da polilinha, com o rumo do trecho."""
    if len(points) == 1:
        return points[0][0], points[0][1], 0.0
    restante = distance_m
    for (lon1, lat1), (lon2, lat2) in itertools.pairwise(points):
        dx = (lon2 - lon1) * 111_320 * math.cos(math.radians((lat1 + lat2) / 2))
        dy = (lat2 - lat1) * 110_540
        trecho = math.hypot(dx, dy)
        rumo = (math.degrees(math.atan2(dx, dy)) + 360) % 360
        if restante <= trecho or trecho == 0:
            t = 0.0 if trecho == 0 else restante / trecho
            return lon1 + (lon2 - lon1) * t, lat1 + (lat2 - lat1) * t, rumo
        restante -= trecho
    lon, lat = points[-1]
    return lon, lat, rumo
