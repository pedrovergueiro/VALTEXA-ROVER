"""Adapters **reais** que não dependem de um hardware específico.

* :class:`LocalDiskStorage` — o disco do computador embarcado (``shutil``);
* :class:`HttpNetwork` — HTTPS com ``urllib`` da stdlib;
* :class:`OnnxCpuAccelerator` — os *providers* que o ONNX Runtime instalado oferece;
* :class:`LinuxSystemMonitor` — CPU, memória e carga lidas do ``/proc`` (qualquer
  Linux embarcado). A ocupação de GPU/NPU depende do fabricante (NVML, tegrastats,
  SDK da NPU) e fica para o adapter do hardware escolhido: aqui sai ``None``.

Nenhum é simulação. Câmera, GPS, IMU e controlador dependem do hardware que a
ValTexa escolher e ficam como interface (:mod:`valtexa_rover.hal.interfaces`) —
ver ``docs/ROVER_RUNTIME.md``.
"""

from __future__ import annotations

import os
import shutil
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

from valtexa_rover.hal.interfaces import (
    DeviceInfo,
    HttpResponse,
    NetworkUnavailableError,
    SystemStats,
)


class LocalDiskStorage:
    """O disco local. ``quota_bytes`` limita o espaço que o Runtime pode usar.

    Com cota, ``usage()`` mede a pasta. Com dezenas de milhares de arquivos isso
    custa (o estresse da Parte 5 mostrou: a missão caía de ~100 para ~20 quadros/s
    medindo a pasta a cada leitura de saúde), então a medida vale ``cache_s``
    segundos. Sem cota, é o ``disk_usage`` do sistema (barato, sem cache).
    """

    def __init__(self, root: Path | str, quota_bytes: int | None = None,
                 cache_s: float = 10.0) -> None:
        self.root = str(root)
        Path(self.root).mkdir(parents=True, exist_ok=True)
        self.quota_bytes = quota_bytes
        self.cache_s = cache_s
        self._medida: tuple[float, int] | None = None

    def usage(self) -> tuple[int, int]:
        total, _used, free = shutil.disk_usage(self.root)
        if self.quota_bytes is None:
            return total, free
        agora = time.monotonic()
        if self._medida is None or agora - self._medida[0] >= self.cache_s:
            self._medida = (agora, _tamanho(self.root))
        usado = self._medida[1]
        return self.quota_bytes, max(0, min(free, self.quota_bytes - usado))

    def invalidate(self) -> None:
        """Esquece a medida (a retenção chama depois de apagar arquivos)."""
        self._medida = None

    def describe(self) -> DeviceInfo:
        return DeviceInfo("armazenamento", "disco local", False, {"cota": self.quota_bytes})


def _pode_ter_enviado(exc: BaseException) -> bool:
    """Se o corpo do pedido pode ter saído antes da falha (o orçamento conta pelo lado seguro).

    Só não sai quando a conexão nem abriu: DNS que falha ou recusa de conexão.
    """
    motivo = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    return not isinstance(motivo, (socket.gaierror, ConnectionRefusedError))


class HttpNetwork:
    """HTTPS até a plataforma. Falha de conexão vira :class:`NetworkUnavailableError`.

    ``is_online`` devolve o resultado da última tentativa: o Runtime não fica
    "pingando" a nuvem — ele tenta quando tem o que enviar.
    """

    def __init__(self, base_url: str, timeout_s: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._online = True

    def is_online(self) -> bool:
        return self._online

    def send(self, method: str, path: str, body: bytes, headers: dict[str, str]) -> HttpResponse:
        request = urllib.request.Request(
            self.base_url + path, data=body or None, method=method, headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                self._online = True
                return HttpResponse(response.status, response.read(), dict(response.headers))
        except urllib.error.HTTPError as exc:
            self._online = True
            return HttpResponse(exc.code, exc.read(), dict(exc.headers or {}))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self._online = False
            raise NetworkUnavailableError(
                str(exc), pode_ter_enviado=_pode_ter_enviado(exc)) from exc

    def describe(self) -> DeviceInfo:
        return DeviceInfo("rede", "HTTPS", False, {"base_url": self.base_url})


class OnnxCpuAccelerator:
    """O que o ONNX Runtime instalado oferece. ``prefer`` ordena (GPU/NPU primeiro).

    É real: lista os *providers* da instalação. Num computador com NPU, o
    adapter do fabricante entra aqui (ex.: ``TensorrtExecutionProvider``,
    ``QNNExecutionProvider``) sem tocar na Edge AI.
    """

    def __init__(self, prefer: list[str] | None = None, name: str = "CPU") -> None:
        self.prefer = prefer or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.name = name

    def providers(self) -> list[str]:
        import onnxruntime as ort

        disponiveis = ort.get_available_providers()
        return [p for p in self.prefer if p in disponiveis] or disponiveis

    def describe(self) -> DeviceInfo:
        return DeviceInfo("acelerador", self.name, False, {"providers": self.providers()})


def _tamanho(raiz: str) -> int:
    """Soma dos arquivos da pasta (``scandir``: um ``stat`` por arquivo, sem ``exists``)."""
    total = 0
    pilha = [raiz]
    while pilha:
        try:
            with os.scandir(pilha.pop()) as entradas:
                for e in entradas:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            pilha.append(e.path)
                        elif e.is_file(follow_symlinks=False):
                            total += e.stat(follow_symlinks=False).st_size
                    except FileNotFoundError:
                        continue  # apagado no meio da medida
        except FileNotFoundError:
            continue
    return total


class LinuxSystemMonitor:
    """CPU, memória e carga do Linux, pelo ``/proc``. Não é simulação.

    A CPU é a fração de tempo não ocioso **entre duas leituras** (como o ``top``): a
    primeira leitura só guarda a referência e devolve ``None``. ``proc`` aponta para
    outra raiz nos testes (um ``/proc`` falso em pasta temporária).
    """

    def __init__(self, proc: Path | str = "/proc") -> None:
        self.proc = Path(proc)
        self._anterior: tuple[int, int] | None = None

    def _cpu(self) -> float | None:
        try:
            linha = (self.proc / "stat").read_text().splitlines()[0]
        except (OSError, IndexError):
            return None
        campos = [int(x) for x in linha.split()[1:]]
        if len(campos) < 4:
            return None
        ocioso = campos[3] + (campos[4] if len(campos) > 4 else 0)  # idle + iowait
        total = sum(campos[:8])
        anterior, self._anterior = self._anterior, (total, ocioso)
        if anterior is None or total <= anterior[0]:
            return None
        delta_total = total - anterior[0]
        delta_ocioso = ocioso - anterior[1]
        return round(100.0 * max(0.0, min(1.0, 1 - delta_ocioso / delta_total)), 1)

    def _memoria(self) -> tuple[int | None, int | None]:
        try:
            linhas = (self.proc / "meminfo").read_text().splitlines()
        except OSError:
            return None, None
        kb = {}
        for linha in linhas:
            nome, _, resto = linha.partition(":")
            partes = resto.split()
            if partes and partes[0].isdigit():
                kb[nome.strip()] = int(partes[0]) * 1024
        total = kb.get("MemTotal")
        livre = kb.get("MemAvailable", kb.get("MemFree"))
        if total is None or livre is None:
            return None, total
        return max(0, total - livre), total

    def _carga(self) -> float | None:
        try:
            return float((self.proc / "loadavg").read_text().split()[0])
        except (OSError, IndexError, ValueError):
            return None

    def read(self) -> SystemStats:
        usada, total = self._memoria()
        return SystemStats(cpu_percent=self._cpu(), memory_used_bytes=usada,
                           memory_total_bytes=total, load_1m=self._carga())

    def describe(self) -> DeviceInfo:
        return DeviceInfo("sistema", "Linux /proc", False,
                          {"acelerador": "não medido (adapter do fabricante)"})
