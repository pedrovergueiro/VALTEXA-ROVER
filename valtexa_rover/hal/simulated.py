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
from dataclasses import replace
from pathlib import Path

import numpy as np
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


class SimulatedRgbdCamera:
    """SIMULAÇÃO: OAK-D de mentira, com foto, mapa de profundidade e captura por região.

    A cena é a saia de uma planta (verde, a ``distancia_mm``) vista de lado, com:

    * ``vaos``: faixas ``(x0, x1)`` da cena por onde aparece o fundo (a ``fundo_mm``, cor
      ``cor_fundo``) — a rua, a linha vizinha;
    * ``manchas``: quadrados ``{"x", "y", "lado", "cor"}`` na cena. ``cor_regiao`` (opcional)
      é a cor que a mancha mostra numa captura por região que a cobre — a exposição certa
      revela a lesão de verdade; ``so_na_primeira`` faz da mancha um reflexo, que só existe
      na primeira captura; ``no_fundo`` põe a mancha no fundo (um objeto vermelho na rua).

    Cada captura desloca a cena ``passo_px`` para a esquerda (o Rover anda). ``cobertura`` é a
    fração de pixels com medida no mapa; ``cobertura_primeira``, a da primeira captura só (a
    OAK-D que perdeu a medida num quadro). O mapa tem 1/5 do lado da foto.
    """

    def __init__(self, camera_id: str, clock: Clock, *, tamanho: tuple[int, int] = (1600, 1200),
                 distancia_mm: float = 270.0, fundo_mm: float = 2500.0,
                 cor_planta: tuple[int, int, int] = (0, 255, 0),
                 cor_fundo: tuple[int, int, int] = (128, 128, 128),
                 vaos: Sequence[tuple[int, int]] = (), manchas: Sequence[dict] = (),
                 passo_px: int = 24, cobertura: float = 1.0,
                 cobertura_primeira: float | None = None, altura_mm: float = 1000.0,
                 hfov_deg: float | None = None, textura: int = 14, seed: int = 0) -> None:
        self.camera_id = camera_id
        self.clock = clock
        self.tamanho = tamanho
        self.distancia_mm = distancia_mm
        self.fundo_mm = fundo_mm
        self.cor_planta = cor_planta
        self.cor_fundo = cor_fundo
        self.vaos = list(vaos)
        self.manchas = [dict(m) for m in manchas]
        self.passo_px = passo_px
        self.cobertura = cobertura
        self.cobertura_primeira = cobertura_primeira
        self.altura_mm = altura_mm
        self.hfov_deg = hfov_deg
        self.rng = np.random.default_rng(seed)
        # Textura de folhagem presa à cena (anda junto com ela): é o que o alinhamento usa.
        bruta = np.random.default_rng(seed + 1).integers(-textura, textura + 1,
                                                         (tamanho[1] // 4, tamanho[0] // 2 + 1))
        self._textura = np.repeat(np.repeat(bruta, 4, axis=0), 4, axis=1)
        self.capturas = 0
        self.regioes: list[tuple[float, float, float, float]] = []
        """As regiões pedidas por :meth:`capture_region`, em ordem."""

    def _render(self, regiao: tuple[float, float, float, float] | None) -> Frame:
        indice = self.capturas
        self.capturas += 1
        largura, altura = self.tamanho
        deslocamento = indice * self.passo_px
        rgb = np.empty((altura, largura, 3), dtype=np.uint8)
        rgb[:] = self.cor_planta
        prof = np.full((altura // 5, largura // 5), self.distancia_mm, dtype=np.float64)
        for x0, x1 in self.vaos:
            a, b = max(0, x0 - deslocamento), min(largura, x1 - deslocamento)
            if b > a:
                rgb[:, a:b] = self.cor_fundo
                prof[:, a // 5:max(a // 5 + 1, b // 5)] = self.fundo_mm
        for m in self.manchas:
            if m.get("so_na_primeira") and indice > 0:
                continue
            x0, y0 = int(m["x"]) - deslocamento, int(m["y"])
            x1, y1 = x0 + int(m["lado"]), y0 + int(m["lado"])
            a, b = max(0, x0), min(largura, x1)
            if b <= a:
                continue
            cor = m["cor"]
            if regiao is not None and m.get("cor_regiao") is not None:
                cx, cy = (x0 + x1) / 2 / largura, (y0 + y1) / 2 / altura
                if regiao[0] <= cx <= regiao[2] and regiao[1] <= cy <= regiao[3]:
                    cor = m["cor_regiao"]
            rgb[y0:y1, a:b] = cor
            if m.get("no_fundo"):
                prof[y0 // 5:max(y0 // 5 + 1, y1 // 5), a // 5:max(a // 5 + 1, b // 5)] = (
                    self.fundo_mm)
        largura_textura = self._textura.shape[1]
        colunas = (np.arange(largura) + deslocamento) % largura_textura
        relevo = self._textura[:altura][:, colunas]
        rgb = np.clip(rgb.astype(np.int16) + relevo[..., None], 0, 255).astype(np.uint8)
        cobertura = (self.cobertura_primeira if indice == 0 and self.cobertura_primeira
                     is not None else self.cobertura)
        if cobertura < 1.0:
            prof[self.rng.random(prof.shape) >= cobertura] = 0.0
        saida = io.BytesIO()
        Image.fromarray(rgb).save(saida, "JPEG", quality=92)
        validos = prof[prof > 0]
        distancia = float(np.median(validos)) if validos.size > prof.size // 20 else None
        return Frame(self.camera_id, indice, self.clock.now(), saida.getvalue(), "image/jpeg",
                     largura, altura, source="rgbd-simulada", distance_mm=distancia,
                     camera_height_mm=self.altura_mm, hfov_deg=self.hfov_deg,
                     depth=prof.astype(np.uint16))

    def capture(self) -> Frame:
        return self._render(None)

    def capture_region(self, region: tuple[float, float, float, float],
                       distance_mm: float | None = None) -> Frame:
        self.regioes.append(tuple(region))
        return replace(self._render(tuple(region)), view="regiao")

    def describe(self) -> DeviceInfo:
        return DeviceInfo("camera", "OAK-D simulada (RGB-D)", True,
                          {"camera_id": self.camera_id, "passo_px": self.passo_px})


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
    """SIMULAÇÃO: controlador que só registra os pedidos de parada e de velocidade."""

    def __init__(self, estado: dict | None = None) -> None:
        self.stops: list[str] = []
        self.velocidades: list[dict] = []
        self.estado_extra = dict(estado or {})
        """O que a navegação publicaria (``modo``, ``speed_ms``, ``reducao_permitida``...)."""

    def state(self) -> dict:
        return {"moving": not self.stops, "stops": len(self.stops), **self.estado_extra}

    def stop(self, reason: str) -> None:
        self.stops.append(reason)

    def pedir_velocidade(self, fator: float, segundos: float, motivo: str) -> dict:
        pedido = {"fator": fator, "seconds": segundos, "reason": motivo}
        self.velocidades.append(pedido)
        return pedido

    def describe(self) -> DeviceInfo:
        return DeviceInfo("controlador", "registro de comandos", True)


class SimulatedThermal:
    """SIMULAÇÃO: câmera térmica que devolve um campo de temperatura com um ponto quente."""

    def __init__(self, base_c: float = 26.0, quente: tuple[float, float, float] | None = None,
                 tamanho: tuple[int, int] = (120, 160), relogio=None, raio_px: int = 5) -> None:
        self.base_c, self.quente, self.tamanho, self.raio_px = base_c, quente, tamanho, raio_px
        self.relogio = relogio
        self.capturas = 0

    def capture(self):
        from datetime import datetime

        from valtexa_rover._compat import UTC
        from valtexa_rover.hal.interfaces import ThermalFrame

        self.capturas += 1
        h, w = self.tamanho
        campo = np.full((h, w), self.base_c, dtype=np.float32)
        if self.quente is not None:  # (x, y, +°C) em fração da foto
            fx, fy, delta = self.quente
            r = self.raio_px
            campo[max(0, int(fy * h) - r):int(fy * h) + r,
                  max(0, int(fx * w) - r):int(fx * w) + r] += delta
        agora = self.relogio.now() if self.relogio else datetime.now(UTC)
        return ThermalFrame(campo, agora)

    def describe(self) -> DeviceInfo:
        return DeviceInfo("termica", "câmera térmica simulada", True, {"base_c": self.base_c})


class SimulatedLight:
    """SIMULAÇÃO: luz auxiliar que guarda os níveis pedidos."""

    def __init__(self, nivel: float = 0.0) -> None:
        self.nivel = nivel
        self.historico: list[float] = []

    def level(self) -> float:
        return self.nivel

    def set_level(self, nivel: float) -> float:
        self.nivel = min(1.0, max(0.0, float(nivel)))
        self.historico.append(self.nivel)
        return self.nivel

    def describe(self) -> DeviceInfo:
        return DeviceInfo("luz", "luz auxiliar simulada", True, {"nivel": self.nivel})


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
