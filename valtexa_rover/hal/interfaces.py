"""Hardware Abstraction Layer — os contratos (brief, seções 20 e 53).

O Runtime fala só com estas interfaces. Quando o hardware definitivo for
escolhido (câmera industrial, GNSS RTK, IMU, NPU...), escreve-se um *adapter*
que cumpre o contrato — nada acima desta camada muda. Ver
``docs/ROVER_RUNTIME.md``, seção "Plugar hardware real".

Regras que todo adapter segue:

* **falha é exceção tipada** (:class:`HardwareUnavailableError`), nunca ``None``
  silencioso: o Health Monitor transforma a exceção em falha ativa na telemetria;
* **horário em UTC** (``datetime`` com fuso), coordenadas WGS84;
* ``describe()`` diz o que o adapter é — e um adapter simulado diz que é
  simulação (``simulated=True``). Nada de simulação passa por hardware real.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable


class HardwareUnavailableError(RuntimeError):
    """O dispositivo não respondeu (desconectado, sem sinal, em falha)."""


class NetworkUnavailableError(ConnectionError):
    """Sem enlace com a nuvem. O Runtime guarda tudo e tenta depois.

    ``pode_ter_enviado``: a falha veio **depois** de a conexão abrir (tempo esgotado à espera
    da resposta, conexão cortada no meio do corpo). Os bytes do pedido podem ter saído pela
    Starlink — o orçamento de dados conta, pelo lado seguro. Recusa de conexão e falha de
    DNS não mandam o corpo (``False``, o padrão).
    """

    def __init__(self, *args: object, pode_ter_enviado: bool = False) -> None:
        super().__init__(*args)
        self.pode_ter_enviado = pode_ter_enviado


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """O que um adapter diz de si — vai para a telemetria e para os metadados."""

    kind: str
    model: str
    simulated: bool
    detail: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Frame:
    """Um quadro capturado. ``data`` são os bytes codificados (JPEG/PNG)."""

    camera_id: str
    frame_index: int
    captured_at: datetime
    data: bytes
    content_type: str
    width: int
    height: int
    source: str = ""
    """De onde veio (arquivo, sensor). Só para diagnóstico."""
    distance_mm: float | None = None
    """Distância até a planta, se a câmera mede profundidade (OAK-D: mediana do mapa de
    profundidade no centro). A Edge AI usa para cortar o quadro em recortes de tamanho
    físico fixo; sem ela, vale a distância padrão do perfil do modelo."""
    camera_height_mm: float | None = None
    """Altura da câmera em relação ao chão (andar do mastro). Diz a que altura da planta
    cada recorte está (pseudocaule, folhas baixas, copa)."""
    side: str | None = None
    """``left``/``right``: o lado do rover para que a câmera olha. A nuvem precisa dele para
    pôr cada recorte na posição certa ao longo da linha (fusão por planta)."""
    hfov_deg: float | None = None
    """Campo de visão horizontal **desta** foto. Com a lente corrigida (:mod:`..lente`), é o
    efetivo da câmera ideal, e substitui o nominal do perfil no corte em recortes."""
    depth: Any = field(default=None, compare=False, repr=False)
    """Mapa de profundidade **alinhado a esta foto** (``numpy`` 2D, mm, 0 = sem medida),
    reduzido (lado maior ~320 px) e com a mesma correção de lente da foto. A Edge AI separa
    planta de fundo com ele (:mod:`valtexa_rover.edge_ai.rgbd`). ``None``: câmera sem
    profundidade, ou o mapa não chegou a tempo."""
    view: str = "principal"
    """``principal`` (a captura do ritmo da missão) ou a estratégia de uma vista extra da
    percepção ativa (``regiao``, ``repeticao``)."""


@dataclass(frozen=True, slots=True)
class GpsFix:
    lat: float
    lon: float
    alt_m: float | None
    fix: str
    """``none``, ``gps``, ``float`` ou ``fixed`` (RTK)."""
    hdop: float | None
    recorded_at: datetime
    heading_deg: float | None = None
    speed_ms: float | None = None


@dataclass(frozen=True, slots=True)
class ImuReading:
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    accel_ms2: tuple[float, float, float]
    recorded_at: datetime
    gyro_dps: tuple[float, float, float] | None = None
    """Velocidade angular (°/s) em x, y, z, se o adapter lê o giroscópio. Sem ela, o ensaio
    de IMU deriva a taxa dos ângulos (menos preciso, e o laudo diz)."""


@dataclass(frozen=True, slots=True)
class SensorReading:
    """Leitura de sensores ambientais/internos, como chave → valor."""

    values: dict[str, float]
    recorded_at: datetime


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status: int
    body: bytes
    headers: dict[str, str] = field(default_factory=dict)

    def json(self):
        import json

        return json.loads(self.body or b"null")


@runtime_checkable
class Camera(Protocol):
    camera_id: str

    def capture(self) -> Frame: ...

    def describe(self) -> DeviceInfo: ...


@runtime_checkable
class ActiveCamera(Protocol):
    """Câmera que sabe olhar de novo para uma região (percepção ativa). Opcional.

    ``region`` é ``(x0, y0, x1, y1)`` em fração da foto (0 a 1). A câmera mede a exposição
    — e, se tiver foco automático, foca — naquela região, tira a foto parada e volta ao
    quadro inteiro. ``distance_mm`` é a distância da planta, para câmeras de foco manual.
    Quem não cumpre este contrato recebe só ``capture()`` de novo (a estratégia
    ``repeticao``)."""

    camera_id: str

    def capture(self) -> Frame: ...

    def capture_region(self, region: tuple[float, float, float, float],
                       distance_mm: float | None = None) -> Frame: ...


@runtime_checkable
class Gps(Protocol):
    def read(self) -> GpsFix: ...

    def describe(self) -> DeviceInfo: ...


@runtime_checkable
class Imu(Protocol):
    def read(self) -> ImuReading: ...

    def describe(self) -> DeviceInfo: ...


@runtime_checkable
class Sensors(Protocol):
    """Temperatura interna, umidade, tensão da bateria — o que o robô tiver."""

    def read(self) -> SensorReading: ...

    def describe(self) -> DeviceInfo: ...


@runtime_checkable
class StorageDevice(Protocol):
    """O disco onde o Runtime grava o banco local e os arquivos."""

    root: str

    def usage(self) -> tuple[int, int]:
        """``(total_bytes, livre_bytes)``."""
        ...

    def describe(self) -> DeviceInfo: ...


@runtime_checkable
class Network(Protocol):
    """Enlace com a nuvem. ``send`` levanta :class:`NetworkUnavailableError` sem enlace.

    O adapter não assina nem interpreta nada: leva bytes e devolve a resposta.
    A assinatura é do :mod:`valtexa_rover.runtime.client`.
    """

    def is_online(self) -> bool: ...

    def send(
        self, method: str, path: str, body: bytes, headers: dict[str, str]
    ) -> HttpResponse: ...

    def describe(self) -> DeviceInfo: ...


@runtime_checkable
class AiAccelerator(Protocol):
    """Onde a inferência roda: CPU, GPU, NPU. Diz os *providers* do ONNX Runtime."""

    def providers(self) -> list[str]: ...

    def describe(self) -> DeviceInfo: ...


@dataclass(frozen=True, slots=True)
class SystemStats:
    """Uso do computador embarcado. ``None`` = o adapter não mede isso (não é zero)."""

    cpu_percent: float | None = None
    memory_used_bytes: int | None = None
    memory_total_bytes: int | None = None
    load_1m: float | None = None
    accelerator_percent: float | None = None
    """Ocupação da GPU/NPU (%). Depende do fabricante (NVML, tegrastats, SDK da NPU)."""
    accelerator_memory_used_bytes: int | None = None
    accelerator_temperature_c: float | None = None


@runtime_checkable
class SystemMonitor(Protocol):
    """CPU, memória e acelerador de IA do computador embarcado (brief, seção 35)."""

    def read(self) -> SystemStats: ...

    def describe(self) -> DeviceInfo: ...


@runtime_checkable
class AuxiliaryLight(Protocol):
    """Iluminação auxiliar da câmera (nível de 0 a 1). Opcional: a OAK-D Pro tem projetor e
    iluminação **infravermelha**, que não ajudam a foto colorida; luz branca é hardware à parte
    — ver ``rover_v3.LuzPorArquivo``."""

    def level(self) -> float: ...

    def set_level(self, nivel: float) -> float: ...

    def describe(self) -> DeviceInfo: ...


@dataclass(frozen=True, slots=True)
class ThermalFrame:
    """Um quadro térmico: temperatura aparente em °C por pixel (radiométrica)."""

    temperatures_c: Any
    """``(H, W)`` em graus Celsius."""
    captured_at: datetime
    emissivity: float = 0.95


@runtime_checkable
class ThermalCamera(Protocol):
    """Câmera térmica opcional: medida complementar da inspeção adaptativa. Não decide: a
    diferença de temperatura da região vai como evidência ao agrônomo, marcada não calibrada
    até haver relação medida entre temperatura e doença no café."""

    def capture(self) -> ThermalFrame | None: ...

    def describe(self) -> DeviceInfo: ...


class Controller(Protocol):
    """Controlador de movimento. O Runtime de percepção só lê o estado e pede parada.

    A contenção no polígono, o retorno por energia e a parada de emergência são
    do controlador/firmware de segurança (ver ``saas/docs/DEVICE_PROTOCOL.md``,
    seção 8) — não deste pacote.
    """

    def state(self) -> dict: ...

    def stop(self, reason: str) -> None: ...

    def describe(self) -> DeviceInfo: ...


@dataclass(slots=True)
class Hardware:
    """O conjunto de adapters que o Runtime recebe. Montado por quem inicia o robô."""

    cameras: list[Camera]
    gps: Gps
    imu: Imu | None
    sensors: Sensors | None
    storage: StorageDevice
    network: Network
    accelerator: AiAccelerator
    controller: Controller | None = None
    hardware_version: str = "desconhecido"
    system: SystemMonitor | None = None
    """Opcional: sem ele a saúde sai sem CPU/memória/acelerador (e diz que não mediu)."""
    luz: AuxiliaryLight | None = None
    termica: ThermalCamera | None = None
    """Opcional: a iluminação auxiliar que a percepção ativa ajusta (:mod:`..runtime.percepcao`)."""

    def describe(self) -> list[DeviceInfo]:
        partes = [c.describe() for c in self.cameras]
        partes += [self.gps.describe(), self.storage.describe(), self.network.describe()]
        partes.append(self.accelerator.describe())
        for opcional in (self.imu, self.sensors, self.controller, self.system):
            if opcional is not None:
                partes.append(opcional.describe())
        return partes

    @property
    def simulated(self) -> bool:
        """Verdadeiro se **qualquer** parte for simulada — e isso vai em cada resultado."""
        return any(info.simulated for info in self.describe())
