"""SIMULADOR DO ROVER (brief, seção 41) — o MESMO Runtime, só a HAL é simulada.

Não é um robô de mentira falando "parecido" com a plataforma: é o
:class:`~valtexa_rover.runtime.core.RoverCore` de verdade (Edge AI pelo ONNX
Runtime, Mission Manager com o comando assinado, Model Manager com hash, buffer
circular, retenção, banco local, Sync Engine, OTA) com adapters **simulados e
marcados como simulação**:

* :class:`CampoSimulado` — um talhão com linhas e plantas, no **espaçamento** do
  talhão (entre linhas e entre plantas), e quais plantas estão doentes ou com
  praga (o "chão de verdade" da simulação);
* :class:`CameraDeCena` — câmera que fotografa a planta mais próxima da posição
  do GPS: folha verde (sadia), avermelhada (doença) ou azulada (praga), com
  ruído; pode ficar indisponível num intervalo;
* :class:`~valtexa_rover.hal.simulated.TrajectoryGps` percorrendo as linhas em
  zigue-zague; IMU, sensores e a rede que cai e volta (:class:`PlanoDeRede`);
* o modelo de demonstração (``modelo_demo.onnx``, 282 bytes: média RGB → 3x4 →
  softmax) — **não é um modelo agronômico**; serve para exercer o caminho.

O protocolo é o do robô real: requisições assinadas (Ed25519) pela HAL de rede —
:class:`~valtexa_rover.hal.local.HttpNetwork` contra uma plataforma no ar
(``python -m valtexa_rover.sim``) ou uma ponte para a API em processo (testes e
``python -m simulator --rover-runtime`` no ``saas/``).
"""

from __future__ import annotations

import io
import itertools
import math
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from valtexa_rover._compat import UTC
from valtexa_rover.clock import Clock, ManualClock
from valtexa_rover.config import RuntimeConfig
from valtexa_rover.hal.interfaces import (
    DeviceInfo,
    Frame,
    GpsFix,
    Hardware,
    HardwareUnavailableError,
    Network,
)
from valtexa_rover.hal.local import LocalDiskStorage, OnnxCpuAccelerator
from valtexa_rover.hal.simulated import (
    FlakyNetwork,
    SimulatedController,
    SimulatedImu,
    SimulatedSensors,
    SimulatedSystemMonitor,
    TrajectoryGps,
)
from valtexa_rover.runtime.client import DeviceIdentity
from valtexa_rover.runtime.core import RoverCore, StepReport
from valtexa_rover.runtime.model_manager import FolderArtifactSource

AQUI = Path(__file__).parent
MODELO_DEMO = AQUI / "modelo_demo.onnx"
MODELO_DEMO_SHA256 = "f29e75543508ecfacd69600e1eddcfafb96b52835ebfce250761b973a5ca345a"
"""Hash do modelo de demonstração — é o que se registra no Model Registry da simulação."""

CORES = {"sadia": (40, 170, 50), "doenca": (200, 40, 30), "praga": (40, 50, 200)}
M_POR_GRAU_LAT = 110_540.0


def instalar_modelo_demo(pasta: Path) -> str:
    """Copia o modelo de demonstração para uma pasta de artefatos (nomeado pelo hash)."""
    pasta.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(MODELO_DEMO, pasta / f"{MODELO_DEMO_SHA256}.onnx")
    shutil.copyfile(MODELO_DEMO.with_suffix(".labels.json"),
                    pasta / f"{MODELO_DEMO_SHA256}.labels.json")
    return MODELO_DEMO_SHA256


@dataclass(frozen=True, slots=True)
class Planta:
    id: str
    linha: int
    x_m: float
    lat: float
    lon: float
    condicao: str = "sadia"


@dataclass
class CampoSimulado:
    """Talhão simulado: ``linhas`` linhas paralelas, plantas a cada ``espacamento_plantas_m``."""

    plantas: list[Planta]
    origem: tuple[float, float]
    espacamento_linhas_m: float
    espacamento_plantas_m: float
    linhas: int

    @classmethod
    def gerar(cls, origem_lon: float, origem_lat: float, *, linhas: int = 3,
              plantas_por_linha: int = 20, espacamento_linhas_m: float = 3.5,
              espacamento_plantas_m: float = 0.7,
              doentes: dict[str, str] | None = None) -> CampoSimulado:
        """Linhas no sentido norte, a partir da origem para o leste. ``doentes``: id → condição."""
        m_lon = 111_320.0 * math.cos(math.radians(origem_lat))
        plantas = []
        for linha in range(linhas):
            for i in range(plantas_por_linha):
                pid = f"L{linha + 1}-P{i + 1:03d}"
                x = i * espacamento_plantas_m
                plantas.append(Planta(
                    id=pid, linha=linha + 1, x_m=x,
                    lat=origem_lat + x / M_POR_GRAU_LAT,
                    lon=origem_lon + linha * espacamento_linhas_m / m_lon,
                    condicao=(doentes or {}).get(pid, "sadia"),
                ))
        return cls(plantas, (origem_lon, origem_lat), espacamento_linhas_m,
                   espacamento_plantas_m, linhas)

    def trajetoria(self) -> list[tuple[float, float]]:
        """Zigue-zague pelas linhas (lon, lat) — o rover anda ao lado de cada linha."""
        pontos = []
        for linha in range(1, self.linhas + 1):
            da_linha = [p for p in self.plantas if p.linha == linha]
            ida = [(p.lon, p.lat) for p in da_linha]
            pontos += ida if linha % 2 else list(reversed(ida))
        return pontos

    def mais_proxima(self, lat: float, lon: float) -> tuple[Planta, float]:
        """A planta mais perto da posição: linha pela longitude, planta pela posição na linha.

        O(1) — o estresse roda centenas de milhares de quadros.
        """
        if not hasattr(self, "_indice"):
            self._indice = {}
            for p in self.plantas:
                self._indice.setdefault(p.linha, []).append(p)
        m_lon = 111_320.0 * math.cos(math.radians(lat))
        origem_lon, origem_lat = self.origem
        linha_f = (lon - origem_lon) * m_lon / self.espacamento_linhas_m
        candidatas = []
        for linha in {math.floor(linha_f) + 1, math.ceil(linha_f) + 1}:
            plantas = self._indice.get(min(max(linha, 1), self.linhas))
            if not plantas:
                continue
            x = (lat - origem_lat) * M_POR_GRAU_LAT
            i = min(max(round(x / self.espacamento_plantas_m), 0), len(plantas) - 1)
            candidatas += plantas[max(0, i - 1):i + 2]
        melhor = min(candidatas, key=lambda p: ((p.lat - lat) * M_POR_GRAU_LAT) ** 2
                     + ((p.lon - lon) * m_lon) ** 2)
        d = math.hypot((melhor.lat - lat) * M_POR_GRAU_LAT, (melhor.lon - lon) * m_lon)
        return melhor, d

    def regiao(self, fix: GpsFix | None) -> dict | None:
        """``region_resolver`` do Core: a planta (e a linha) diante do rover."""
        if fix is None:
            return None
        planta, d = self.mais_proxima(fix.lat, fix.lon)
        if d > max(self.espacamento_plantas_m, 1.0):
            return None
        return {"plant_id": planta.id, "row": planta.linha, "x_m": round(planta.x_m, 3),
                "distance_m": round(d, 3), "source": "simulador (mapa de plantas)"}


class CameraDeCena:
    """SIMULAÇÃO: fotografa a planta mais próxima do GPS. Cor = condição da planta."""

    def __init__(self, camera_id: str, campo: CampoSimulado, gps: TrajectoryGps, clock: Clock,
                 *, size: tuple[int, int] = (320, 240), seed: int = 0,
                 indisponivel: Callable[[int], bool] | None = None,
                 marca_nuvem: dict[str, str] | None = None) -> None:
        self.camera_id = camera_id
        self.marca_nuvem = marca_nuvem
        self.campo = campo
        self.gps = gps
        self.clock = clock
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.indisponivel = indisponivel
        self._index = 0

    def capture(self) -> Frame:
        indice = self._index
        self._index += 1
        if self.indisponivel and self.indisponivel(indice):
            raise HardwareUnavailableError(f"{self.camera_id}: câmera indisponível (simulação)")
        fix = self.gps.read()
        planta, _d = self.campo.mais_proxima(fix.lat, fix.lon)
        largura, altura = self.size
        rgb = np.zeros((altura, largura, 3), dtype=np.int16) + np.array(CORES[planta.condicao])
        rgb += self.rng.integers(-35, 36, rgb.shape)
        saida = io.BytesIO()
        Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(saida, "JPEG", quality=90)
        if self.marca_nuvem and planta.condicao in self.marca_nuvem:
            # SIMULAÇÃO: o "chão de verdade" vai no fim do JPEG para o analisador
            # simulado da nuvem (``VTXSIM:<rótulo>``, ver saas ai_pipeline). O JPEG
            # continua válido; a Edge AI não lê a marca (decodifica só a imagem).
            saida.write(b"VTXSIM:" + self.marca_nuvem[planta.condicao].encode())
        return Frame(self.camera_id, indice, self.clock.now(), saida.getvalue(), "image/jpeg",
                     largura, altura, source=f"simulador:{planta.id}")

    def describe(self) -> DeviceInfo:
        return DeviceInfo("camera", "câmera de cena", True, {"camera_id": self.camera_id})


@dataclass
class PlanoDeRede:
    """Janelas sem rede, em passos da missão: ``[(início, fim), ...]`` (fim exclusivo)."""

    sem_rede: list[tuple[int, int]] = field(default_factory=list)

    def online(self, passo: int) -> bool:
        return not any(a <= passo < b for a, b in self.sem_rede)


@dataclass
class ResultadoDaMissao:
    passos: int = 0
    quadros: int = 0
    eventos: int = 0
    falhas_de_camera: int = 0
    passos_offline: int = 0
    reinicios: int = 0
    categorias: dict[str, int] = field(default_factory=dict)


class RoverSimulado:
    """Monta o Runtime de verdade sobre a HAL simulada e conduz a missão passo a passo."""

    def __init__(self, dados: Path | str, identidade: DeviceIdentity, transporte,
                 campo: CampoSimulado, *, clock: Clock | None = None, cameras: int = 2,
                 velocidade_ms: float = 0.7,
                 config: RuntimeConfig | None = None, seed: int = 0,
                 camera_indisponivel: Callable[[str, int], bool] | None = None,
                 artefatos: Path | None = None, marca_nuvem: dict[str, str] | None = None,
                 **core_kwargs) -> None:
        self.dados = Path(dados)
        self.identidade = identidade
        self.campo = campo
        self.clock = clock or ManualClock(datetime.now(UTC))
        self.gps = TrajectoryGps(campo.trajetoria(), self.clock, speed_ms=velocidade_ms)
        if isinstance(transporte, FlakyNetwork):
            self.rede = transporte
        elif hasattr(transporte, "send"):
            self.rede = _Envoltorio(transporte)  # HttpNetwork de verdade
        else:
            self.rede = FlakyNetwork(transporte)  # função (ponte para a API em processo)
        self.cameras = [
            CameraDeCena(f"cam-{i}", campo, self.gps, self.clock, seed=seed + i,
                         indisponivel=(lambda n, cid=f"cam-{i}": camera_indisponivel(cid, n))
                         if camera_indisponivel else None, marca_nuvem=marca_nuvem)
            for i in range(cameras)
        ]
        self.hardware = Hardware(
            cameras=self.cameras, gps=self.gps, imu=SimulatedImu(self.clock, self.gps),
            sensors=SimulatedSensors(self.clock), storage=LocalDiskStorage(self.dados),
            network=self.rede, accelerator=OnnxCpuAccelerator(),
            controller=SimulatedController(), hardware_version="Rover Edge V1 (SIMULADO)",
            system=SimulatedSystemMonitor(),
        )
        self.artefatos = artefatos or self.dados / "artefatos"
        instalar_modelo_demo(self.artefatos)
        self.config = config or RuntimeConfig()
        self.core_kwargs = {"telemetry_interval_s": 5.0, "seed": seed, **core_kwargs}
        self.core = self.novo_core()

    def novo_core(self) -> RoverCore:
        core = RoverCore(self.hardware, self.dados, self.identidade, config=self.config,
                         clock=self.clock, artifact_source=FolderArtifactSource(self.artefatos),
                         region_resolver=self.campo.regiao, **self.core_kwargs)
        core.boot()
        return core

    def reiniciar(self) -> None:
        """Simula o robô desligado e ligado de novo (o banco e os arquivos ficam)."""
        self.core.close()
        self.core = self.novo_core()

    def rodar(self, passos: int, plano: PlanoDeRede | None = None, *, inicio: int = 0,
              reiniciar_em: set[int] | None = None, dt_s: float = 1.0) -> ResultadoDaMissao:
        res = ResultadoDaMissao()
        plano = plano or PlanoDeRede()
        for n in range(inicio, inicio + passos):
            if reiniciar_em and n in reiniciar_em:
                self.reiniciar()
                res.reinicios += 1
            online = plano.online(n)
            self.rede.set_online(online)
            res.passos_offline += 0 if online else 1
            rel: StepReport = self.core.step()
            res.passos += 1
            res.quadros += rel.frames
            res.eventos += rel.events
            res.falhas_de_camera += rel.camera_failures
            for k, v in rel.categories.items():
                res.categorias[k] = res.categorias.get(k, 0) + v
            self.clock.advance(dt_s)
        return res

    def sincronizar(self, max_passos: int = 500) -> int:
        """Rede de volta: sincroniza até esvaziar a fila. Devolve o que ainda falta."""
        self.rede.set_online(True)
        for _ in range(max_passos):
            if self.core.sync.pending() == 0 and not self._telemetria_pendente():
                break
            self.core.sync.step(200)
            self.clock.advance(1.0)
        return self.core.sync.pending()

    def _telemetria_pendente(self) -> int:
        return self.core.store.scalar("SELECT COUNT(*) FROM telemetry WHERE sent = 0") or 0


class ControladorDeRota(SimulatedController):
    """SIMULAÇÃO: diz ``route_done`` quando o GPS chega ao fim da trajetória."""

    def __init__(self, gps: TrajectoryGps) -> None:
        super().__init__()
        self.gps = gps
        pontos = gps.waypoints
        self.comprimento_m = sum(
            math.hypot((b[0] - a[0]) * 111_320 * math.cos(math.radians(a[1])),
                       (b[1] - a[1]) * M_POR_GRAU_LAT)
            for a, b in itertools.pairwise(pontos))

    def state(self) -> dict:
        andado = (self.gps.clock.monotonic() - self.gps._start) * self.gps.speed_ms
        return {**super().state(), "route_done": andado >= self.comprimento_m}


def hal_de_demonstracao(*, dados: Path, url: str, clock: Clock) -> Hardware:
    """HAL SIMULADA para ``valtexa-rover servico --hal valtexa_rover.sim:hal_de_demonstracao``.

    Campo de 3 linhas x 30 plantas com 3 doentes, duas câmeras de cena, GPS na rota,
    rede HTTPS de verdade até ``url``. Para demonstração e homologação — nunca em campo.
    """
    from valtexa_rover.hal.local import HttpNetwork

    campo = CampoSimulado.gerar(-47.06, -22.90, linhas=3, plantas_por_linha=30,
                                doentes={"L1-P010": "doenca", "L2-P020": "praga",
                                         "L3-P005": "doenca"})
    gps = TrajectoryGps(campo.trajetoria(), clock, speed_ms=0.7)
    instalar_modelo_demo(Path(dados) / "ota" / "models")
    return Hardware(
        cameras=[CameraDeCena(f"cam-{i}", campo, gps, clock, seed=i) for i in range(2)],
        gps=gps, imu=SimulatedImu(clock, gps), sensors=SimulatedSensors(clock),
        storage=LocalDiskStorage(dados), network=HttpNetwork(url),
        accelerator=OnnxCpuAccelerator(), controller=ControladorDeRota(gps),
        hardware_version="Rover Edge V1 (SIMULADO)", system=SimulatedSystemMonitor(),
    )


class _Envoltorio(FlakyNetwork):
    """Uma rede real (``HttpNetwork``) com a queda simulada por cima."""

    def __init__(self, rede: Network) -> None:
        super().__init__(rede.send)


__all__ = [
    "MODELO_DEMO_SHA256",
    "CameraDeCena",
    "CampoSimulado",
    "ControladorDeRota",
    "PlanoDeRede",
    "Planta",
    "ResultadoDaMissao",
    "RoverSimulado",
    "hal_de_demonstracao",
    "instalar_modelo_demo",
]
