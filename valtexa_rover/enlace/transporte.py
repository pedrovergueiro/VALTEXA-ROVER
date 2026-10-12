"""Transportes: cada enlace de rádio é uma interface IP, e o protocolo fala UDP por ela.

A maioria dos rádios industriais (sub-GHz, 5 GHz, malha, 4G) é **ponte Ethernet/IP**:
o software não fala o protocolo do rádio; manda datagramas UDP para o IP do outro lado
através da interface daquele rádio. :class:`TransporteUdp` faz isso (``interface`` usa
``SO_BINDTODEVICE``, que pede ``CAP_NET_RAW``; sem ela, a rota do sistema decide pela
sub-rede de cada rádio — o caminho recomendado no ``hardware.json``).

O emulador (:mod:`.emulador`) implementa a mesma interface em processo.
"""

from __future__ import annotations

import contextlib
import errno
import socket
import time
from typing import Any, Protocol


class Transporte(Protocol):
    nome: str
    mtu: int

    def enviar(self, dados: bytes, destino: Any) -> None: ...

    def receber(self) -> list[tuple[bytes, Any]]:
        """Tudo o que chegou até agora (não bloqueia)."""
        ...

    def fileno(self) -> int | None: ...

    def fechar(self) -> None: ...


class Relogio:
    """Hora real: ``agora`` monotônico (timers), ``parede`` em segundos desde 1970."""

    def agora(self) -> float:
        return time.monotonic()

    def parede(self) -> float:
        return time.time()


class RelogioVirtual:
    """Relógio dos testes: só anda quando mandam (caos determinístico e rápido)."""

    def __init__(self, inicio: float = 1_800_000_000.0) -> None:
        self.t = 0.0
        self._base = inicio

    def agora(self) -> float:
        return self.t

    def parede(self) -> float:
        return self._base + self.t

    def avanca(self, dt: float) -> None:
        self.t += dt


class RelogioDeslocado:
    """O mesmo tempo monotônico, com a hora de parede errada por ``desvio_s`` (testes do
    relógio do Rover atrasado ou adiantado)."""

    def __init__(self, base, desvio_s: float) -> None:
        self.base = base
        self.desvio_s = desvio_s

    def agora(self) -> float:
        return self.base.agora()

    def parede(self) -> float:
        return self.base.parede() + self.desvio_s


def endereco(texto: str) -> tuple[str, int]:
    """``"10.20.0.1:47800"`` → ``("10.20.0.1", 47800)``."""
    host, _, porta = texto.rpartition(":")
    return host.strip("[]"), int(porta)


class TransporteUdp:
    """UDP real. ``bind`` = endereço local; ``interface`` opcional (``SO_BINDTODEVICE``)."""

    def __init__(self, nome: str, bind: tuple[str, int], *, mtu: int = 1200,
                 interface: str | None = None, buffer_bytes: int = 4 * 1024 * 1024) -> None:
        self.nome = nome
        self.mtu = mtu
        self.erros_envio = 0
        familia = socket.AF_INET6 if ":" in bind[0] else socket.AF_INET
        self._sock = socket.socket(familia, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        for opcao in (socket.SO_RCVBUF, socket.SO_SNDBUF):
            with contextlib.suppress(OSError):
                self._sock.setsockopt(socket.SOL_SOCKET, opcao, buffer_bytes)
        if interface:
            self._sock.setsockopt(socket.SOL_SOCKET, 25, interface.encode())  # SO_BINDTODEVICE
        self._sock.bind(bind)
        self._sock.setblocking(False)

    @property
    def local(self) -> tuple[str, int]:
        return self._sock.getsockname()[:2]

    def enviar(self, dados: bytes, destino: Any) -> None:
        try:
            self._sock.sendto(dados, destino)
        except OSError as exc:  # rádio fora (rota sumiu, interface caída): é perda
            if exc.errno not in (errno.EAGAIN, errno.ENETUNREACH, errno.EHOSTUNREACH,
                                 errno.ENETDOWN, errno.ECONNREFUSED, errno.EPERM,
                                 errno.EINVAL, errno.EADDRNOTAVAIL):
                raise
            self.erros_envio += 1

    def receber(self) -> list[tuple[bytes, Any]]:
        saida = []
        while True:
            try:
                dados, origem = self._sock.recvfrom(65535)
            except (BlockingIOError, InterruptedError):
                return saida
            except ConnectionRefusedError:
                continue
            saida.append((dados, origem[:2]))

    def fileno(self) -> int | None:
        return self._sock.fileno()

    def fechar(self) -> None:
        self._sock.close()
