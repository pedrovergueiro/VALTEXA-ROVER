"""Ensaios automáticos do Rover — o comando assinado ``ensaio`` do centro de comando.

Cada ensaio mede uma peça e devolve um laudo com critérios e amostras brutas (ver
:mod:`.base`). Roda numa thread (:class:`ExecutorDeEnsaios`), um por vez, com progresso
e cancelamento; o laço do Runtime continua (telemetria, comandos, controle remoto).

Catálogo (:data:`CATALOGO`): ``calor``, ``cameras``, ``profundidade``, ``gnss``, ``imu``,
``rede``, ``energia``, ``relogio`` (FASE 0); ``parada``, ``homem_morto`` (FASE 1). A FASE 5
(validação agronômica) é calculada na plataforma, a partir da validação de campo. Os
ensaios ``enlace_*`` (rádio da fazenda) foram arquivados com a base em 20/10/2026
(``archive/legacy/radio-fazenda/``).
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable
from typing import Any

from valtexa_rover.clock import Clock
from valtexa_rover.ensaios.base import (
    Contexto,
    Ensaio,
    EnsaioCancelado,
    EnsaioIndisponivelError,
    ParametroInvalidoError,
    rodar,
)
from valtexa_rover.ensaios.calor import Calor
from valtexa_rover.ensaios.cameras import Cameras
from valtexa_rover.ensaios.energia import Energia, SensoresComoEnergia
from valtexa_rover.ensaios.gnss import Gnss
from valtexa_rover.ensaios.imu import Imu
from valtexa_rover.ensaios.profundidade import Profundidade
from valtexa_rover.ensaios.rede import Rede
from valtexa_rover.ensaios.relogio import FonteDoRelogio, Relogio
from valtexa_rover.ensaios.seguranca import HomemMorto, Parada
from valtexa_rover.hal.interfaces import Hardware

__all__ = ["CATALOGO", "FASE0", "Contexto", "Ensaio", "EnsaioCancelado",
           "EnsaioIndisponivelError", "ExecutorDeEnsaios", "ParametroInvalidoError",
           "catalogo", "fontes_do_runtime", "rodar", "rodar_por_nome"]

CATALOGO: dict[str, type[Ensaio]] = {
    e.nome: e for e in (Calor, Cameras, Profundidade, Gnss, Imu, Rede, Energia, Relogio, Parada,
                        HomemMorto)
}
FASE0 = ("cameras", "gnss", "imu", "rede", "calor", "energia")
"""Os ensaios da FASE 0 que rodam sozinhos (profundidade pede a trena do operador)."""


def catalogo() -> list[dict[str, Any]]:
    return [e.descreve() for e in CATALOGO.values()]


def rodar_por_nome(nome: str, params: dict | None, ctx: Contexto) -> dict[str, Any]:
    if nome not in CATALOGO:
        raise ParametroInvalidoError(f"ensaio {nome!r}: use {', '.join(CATALOGO)}")
    return rodar(CATALOGO[nome](), params, ctx)


class _TermicaDoMonitor:
    """``termica`` a partir do monitor do sistema: o ``tegrastats`` na Jetson; fora dela,
    só a temperatura do acelerador (sem clocks)."""

    def __init__(self, monitor) -> None:
        self.monitor = monitor

    def ler(self) -> dict[str, Any]:
        bruto = {}
        if hasattr(self.monitor, "ultima_tegrastats"):
            bruto = self.monitor.ultima_tegrastats() or {}
        temps = list((bruto.get("temperaturas_c") or {}).values())
        stats = self.monitor.read()
        temp = max(temps) if temps else stats.accelerator_temperature_c
        return {"temp_c": temp, "cpu_freq_mhz": bruto.get("cpu_freq_mhz"),
                "gpu_freq_mhz": bruto.get("gpu_freq_mhz"),
                "gpu_pct": bruto.get("accelerator_percent", stats.accelerator_percent)}


class _CargaDoMotor:
    """Uma inferência do modelo ativo sobre uma foto real da primeira câmera (tirada na
    primeira chamada e reaproveitada: mede a IA, não a câmera)."""

    def __init__(self, core) -> None:
        self.core = core
        self._quadro = None

    def __call__(self) -> int:
        if self._quadro is None:
            self._quadro = self.core.hardware.cameras[0].capture()
        q = self._quadro
        self.core.engine.infer(q.data, q.distance_mm, q.camera_height_mm, q.hfov_deg)
        return 1


class _RedeDoRobo:
    """Ida e volta até a plataforma pelo canal assinado (``/v1/device/ensaio/eco``)."""

    CAMINHO = "/v1/device/ensaio/eco"

    def __init__(self, client) -> None:
        self.client = client

    def ping(self) -> bool:
        return self.client.request("GET", self.CAMINHO).status == 200

    def upload(self, n: int) -> int:
        r = self.client.request("POST", self.CAMINHO, raw=b"\0" * n,
                                content_type="application/octet-stream")
        return int((r.json() or {}).get("recebidos", 0)) if r.status == 200 else 0

    def download(self, n: int) -> int:
        r = self.client.request("GET", f"{self.CAMINHO}?bytes={n}")
        return len(r.body) if r.status == 200 else 0


def fontes_do_runtime(core) -> dict[str, Any]:
    """As fontes extras de um Runtime de verdade (o que existir neste robô)."""
    fontes: dict[str, Any] = {"rede": _RedeDoRobo(core.client),
                              "relogio": FonteDoRelogio(core)}
    if core.engine.loaded and core.hardware.cameras:
        fontes["carga"] = _CargaDoMotor(core)
    if core.hardware.system is not None:
        fontes["termica"] = _TermicaDoMonitor(core.hardware.system)
    sensores = core.hardware.sensors
    if sensores is not None:
        with contextlib.suppress(Exception):
            if "battery_percent" in sensores.read().values:
                fontes["energia"] = SensoresComoEnergia(sensores)
    return fontes


AoProgresso = Callable[[str, float, str, dict | None], None]
AoTerminar = Callable[[str, dict], None]


class ExecutorDeEnsaios:
    """Um ensaio por vez, numa thread. ``iniciar`` valida antes (erro de parâmetro volta
    na hora); o laudo chega por ``ao_terminar``; o progresso, no máximo a cada
    ``intervalo_progresso_s``, por ``ao_progresso``."""

    def __init__(self, hardware: Hardware, clock: Clock,
                 fontes: dict[str, Any] | Callable[[], dict[str, Any]], *,
                 ao_progresso: AoProgresso | None = None, ao_terminar: AoTerminar | None = None,
                 intervalo_progresso_s: float = 2.0) -> None:
        self.hardware = hardware
        self.clock = clock
        self._fontes = fontes
        self.ao_progresso = ao_progresso
        self.ao_terminar = ao_terminar
        self.intervalo_progresso_s = intervalo_progresso_s
        self._lock = threading.Lock()
        self._atual: tuple[str, threading.Event, threading.Thread] | None = None
        self.ultimo_laudo: dict | None = None

    @property
    def ocupado(self) -> str | None:
        """O id do ensaio em curso, ou ``None``."""
        with self._lock:
            if self._atual is not None and self._atual[2].is_alive():
                return self._atual[0]
            return None

    def iniciar(self, ensaio_id: str, nome: str, params: dict | None) -> dict[str, Any]:
        if nome not in CATALOGO:
            raise ParametroInvalidoError(f"ensaio {nome!r}: use {', '.join(CATALOGO)}")
        classe = CATALOGO[nome]
        validados = classe.valida(params)
        em_curso = self.ocupado
        if em_curso is not None:
            raise EnsaioIndisponivelError(f"já há um ensaio em curso ({em_curso})")
        fontes = self._fontes() if callable(self._fontes) else self._fontes
        cancelar = threading.Event()
        ultimo = [-1e18]

        def progresso(fracao: float, etapa: str, parcial: dict | None) -> None:
            agora = self.clock.monotonic()
            if self.ao_progresso is None:
                return
            if agora - ultimo[0] >= self.intervalo_progresso_s or fracao >= 1.0:
                ultimo[0] = agora
                with contextlib.suppress(Exception):  # progresso perdido não para o ensaio
                    self.ao_progresso(ensaio_id, fracao, etapa, parcial)

        ctx = Contexto(self.hardware, self.clock, fontes, progresso, cancelar)

        def corpo() -> None:
            laudo = rodar(classe(), validados, ctx)
            self.ultimo_laudo = laudo
            if self.ao_terminar is not None:
                self.ao_terminar(ensaio_id, laudo)

        thread = threading.Thread(target=corpo, name=f"ensaio-{nome}", daemon=True)
        with self._lock:
            self._atual = (ensaio_id, cancelar, thread)
        thread.start()
        return {"ensaio": nome, "params": validados, "iniciado": True}

    def cancelar(self, ensaio_id: str) -> bool:
        with self._lock:
            if self._atual is None or self._atual[0] != ensaio_id:
                return False
            if not self._atual[2].is_alive():
                return False
            self._atual[1].set()
            return True

    def aguardar(self, timeout: float | None = None) -> bool:
        with self._lock:
            atual = self._atual
        if atual is None:
            return True
        atual[2].join(timeout)
        return not atual[2].is_alive()
