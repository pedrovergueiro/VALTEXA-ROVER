"""Health Monitor e Telemetry (brief, seções 21 e 25).

O Health Monitor junta, a cada amostra: armazenamento (total/livre, da HAL),
temperatura e bateria (sensores), horímetro (persistido — só sobe), latência e
FPS da Edge AI, e as **falhas ativas** (câmera que não respondeu, GPS sem sinal,
disco abaixo do mínimo, temperatura alta, modelo ausente). A telemetria vira
amostra com ``sequence_number`` do contador monotônico do robô e sobe pelo
``/v1/device/sync`` — os campos são os que a frota já lê (Parte 3):
``storage_total_bytes``, ``storage_free_bytes``, ``operating_hours``, ``faults``.

Parte 7 (observabilidade, seção 35): cada amostra leva também ``health``, a saúde
**padronizada** que a plataforma guarda por robô e usa nos alertas — CPU, memória
e acelerador (do :class:`~valtexa_rover.hal.interfaces.SystemMonitor`, se houver),
disco, temperatura, bateria, cada câmera, GPS/IMU/sensores, rede, fila de
sincronização e Edge AI. O que não foi medido sai ``None`` (não zero), e
``simulated`` lista as partes da HAL que são simulação.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from valtexa_rover.clock import Clock
from valtexa_rover.config import HealthConfig
from valtexa_rover.edge_ai.engine import EdgeAIEngine
from valtexa_rover.hal.interfaces import GpsFix, Hardware, HardwareUnavailableError, SystemStats
from valtexa_rover.runtime.store import LocalStore


class HealthMonitor:
    def __init__(
        self, hardware: Hardware, store: LocalStore, engine: EdgeAIEngine,
        config: HealthConfig, clock: Clock,
    ) -> None:
        self.hardware = hardware
        self.store = store
        self.engine = engine
        self.config = config
        self.clock = clock
        self.faults: dict[str, str] = {}
        self._last_tick = clock.monotonic()
        self.sync_status: Callable[[], dict] | None = None
        """Ligado pelo Core ao Sync Engine (que nasce depois do Health Monitor)."""
        self.missao_em_curso: Callable[[], bool] = lambda: False
        """Ligado pelo Core: sem missão, robô sem modelo carregado não está com defeito."""
        self.modelo_indisponivel: str | None = None
        """Por que a pré-carga não conseguiu nenhum modelo designado (``None``: conseguiu
        ou não há designação). Com isso o robô não começaria a missão: é falha."""

    def fault(self, code: str, detail: str) -> bool:
        """Liga uma falha. Devolve ``True`` se ela é nova (o Core emite evento só então)."""
        nova = code not in self.faults
        self.faults[code] = detail
        return nova

    def clear(self, code: str) -> None:
        self.faults.pop(code, None)

    def tick(self) -> None:
        """Soma o tempo ligado ao horímetro (persistido)."""
        agora = self.clock.monotonic()
        delta = max(0.0, agora - self._last_tick)
        self._last_tick = agora
        if delta:
            self.store.put("operating_seconds", float(self.store.get("operating_seconds", 0.0))
                           + delta)

    def snapshot(self) -> dict:
        self.tick()
        total, livre = self.hardware.storage.usage()
        if livre < self.config.min_free_bytes:
            self.fault("storage_low", f"{livre} bytes livres (mínimo {self.config.min_free_bytes})")
        else:
            self.clear("storage_low")
        sensores: dict[str, float] = {}
        if self.hardware.sensors is not None:
            try:
                sensores = dict(self.hardware.sensors.read().values)
                self.clear("sensors")
            except HardwareUnavailableError as exc:
                self.fault("sensors", str(exc))
        temperatura = sensores.get("temperature_c")
        if temperatura is not None and temperatura > self.config.max_temperature_c:
            self.fault("temperature_high", f"{temperatura} °C")
        else:
            self.clear("temperature_high")
        # Sem modelo só é falha quando atrapalha: missão rodando sem modelo, ou talhão
        # designado cujo modelo não carrega (a próxima missão não começaria). Robô parado,
        # esperando designação, está disponível — não em erro.
        if self.engine.loaded:
            self.clear("model_missing")
        elif self.missao_em_curso():
            self.fault("model_missing", "missão em curso sem modelo do registry carregado")
        elif self.modelo_indisponivel:
            self.fault("model_missing", self.modelo_indisponivel)
        else:
            self.clear("model_missing")
        return {
            "storage_total_bytes": total,
            "storage_free_bytes": livre,
            "operating_hours": round(float(self.store.get("operating_seconds", 0.0)) / 3600, 4),
            "temperature_c": temperatura,
            "battery_percent": sensores.get("battery_percent"),
            "edge_ai_fps": (
                None if self.engine.stats.fps is None else round(self.engine.stats.fps, 2)
            ),
            "edge_ai_latency_ms": (
                None if self.engine.stats.mean_latency_ms is None
                else round(self.engine.stats.mean_latency_ms, 2)
            ),
            "faults": sorted(self.faults),
            "fault_detail": dict(self.faults),
        }


    def _sistema(self) -> SystemStats | None:
        if self.hardware.system is None:
            return None
        try:
            stats = self.hardware.system.read()
            self.clear("system")
            return stats
        except (HardwareUnavailableError, OSError) as exc:
            self.fault("system", str(exc))
            return None

    def relatorio(self, online: bool, saude: dict | None = None) -> dict:
        """A saúde padronizada (o campo ``health`` da telemetria)."""
        saude = saude if saude is not None else self.snapshot()
        stats = self._sistema()
        memoria_pct = None
        if stats and stats.memory_used_bytes is not None and stats.memory_total_bytes:
            memoria_pct = round(100 * stats.memory_used_bytes / stats.memory_total_bytes, 1)
        total, livre = saude["storage_total_bytes"], saude["storage_free_bytes"]
        disco_pct = round(100 * (total - livre) / total, 1) if total else None
        acelerador = self.hardware.accelerator.describe()
        cameras = [
            {"id": c.camera_id, "ok": f"camera:{c.camera_id}" not in self.faults,
             "detail": self.faults.get(f"camera:{c.camera_id}")}
            for c in self.hardware.cameras
        ]

        def estado(parte, codigo: str) -> str:
            if parte is None:
                return "ausente"
            return "falha" if codigo in self.faults else "ok"

        sincronizacao = None
        if self.sync_status is not None:
            try:
                sincronizacao = self.sync_status()
            except Exception as exc:  # a saúde nunca derruba a telemetria
                sincronizacao = {"error": f"{type(exc).__name__}: {exc}"}
        return {
            "cpu_percent": stats.cpu_percent if stats else None,
            "memory_percent": memoria_pct,
            "memory_used_bytes": stats.memory_used_bytes if stats else None,
            "memory_total_bytes": stats.memory_total_bytes if stats else None,
            "load_1m": stats.load_1m if stats else None,
            "accelerator": {
                "name": acelerador.model,
                "providers": list(self.hardware.accelerator.providers()),
                "percent": stats.accelerator_percent if stats else None,
                "memory_used_bytes": stats.accelerator_memory_used_bytes if stats else None,
                "temperature_c": stats.accelerator_temperature_c if stats else None,
            },
            "disk_percent": disco_pct,
            "temperature_c": saude["temperature_c"],
            "battery_percent": saude["battery_percent"],
            "cameras": cameras,
            "sensors": {
                "gps": estado(self.hardware.gps, "gps"),
                "imu": "ausente" if self.hardware.imu is None else "nao_monitorado",
                "ambiente": estado(self.hardware.sensors, "sensors"),
            },
            "network": {"online": bool(online)},
            "sync": sincronizacao,
            "edge_ai": {"fps": saude["edge_ai_fps"], "latency_ms": saude["edge_ai_latency_ms"],
                        "model_loaded": self.engine.loaded},
            "simulated": sorted({i.kind for i in self.hardware.describe() if i.simulated}),
        }


class Telemetry:
    """Grava amostras de telemetria na tabela local; o Sync Engine as envia."""

    def __init__(self, store: LocalStore, health: HealthMonitor, clock: Clock) -> None:
        self.store = store
        self.health = health
        self.clock = clock

    def record(self, gps: GpsFix | None, mission_id: str | None, online: bool) -> int:
        saude = self.health.snapshot()
        with self.store.transaction() as conn:
            seq = self.store.next_counter("protocol_sequence")
            amostra = {
                "sequence_number": seq,
                "recorded_at": self.clock.now().isoformat(),
                "mission_id": mission_id,
                "lat": gps.lat if gps else None,
                "lon": gps.lon if gps else None,
                "alt_m": gps.alt_m if gps else None,
                "heading_deg": gps.heading_deg if gps else None,
                "speed_ms": gps.speed_ms if gps else None,
                "rtk_fix": gps.fix if gps else "none",
                "hdop": gps.hdop if gps else None,
                "battery_percent": saude["battery_percent"],
                "temperature_c": saude["temperature_c"],
                "mission_state": "running" if mission_id else "idle",
                "link_quality": 1.0 if online else 0.0,
                "storage_total_bytes": saude["storage_total_bytes"],
                "storage_free_bytes": saude["storage_free_bytes"],
                "operating_hours": saude["operating_hours"],
                "faults": saude["faults"],
                "sensors": {"edge_ai_fps": saude["edge_ai_fps"],
                            "edge_ai_latency_ms": saude["edge_ai_latency_ms"]},
                "health": self.health.relatorio(online, saude),
            }
            conn.execute(
                "INSERT INTO telemetry (sequence_number, recorded_at, payload) VALUES (?, ?, ?)",
                (seq, amostra["recorded_at"], json.dumps(amostra)),
            )
        return seq
