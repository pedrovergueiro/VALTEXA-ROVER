"""Emulador de rádio — **simulação**: nenhum número daqui é medida de rádio de verdade.

Modelo de canal (:class:`ModeloDeCanal`), por sentido:

* **banda** (bit/s) com fila de serialização limitada (``fila_s``: o que passaria disso é
  descartado, como no buffer de um rádio);
* **atraso** fixo + **variação** (uniforme);
* **perda** independente e **perda em rajada** (Gilbert-Elliott: estados bom/ruim com
  probabilidade de transição por pacote);
* **reordenação** (atraso extra aleatório num pacote), **duplicação**, **corrupção**
  (troca bytes; o CRC tem de pegar);
* **quedas** programadas (janelas ``(início, fim)`` em segundos do relógio, 5 a 60 s) e
  **intermitente** (``(ligado_s, desligado_s)`` periódico).

Duas formas:

* **em processo** (:class:`RedeEmulada`): pontos com a interface de
  :class:`~.transporte.Transporte`, relógio virtual, sementes fixas — os testes de caos
  rodam horas de enlace em segundos e sempre do mesmo jeito;
* **por UDP** (:class:`ReleUdp`, ``python -m valtexa_rover.enlace.emulador``): um relé
  entre dois processos de verdade, em tempo real.
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import random
import socket
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any


@dataclass
class ModeloDeCanal:
    banda_bps: float = 1_000_000.0
    atraso_s: float = 0.02
    variacao_s: float = 0.0
    perda: float = 0.0
    # Gilbert-Elliott (desligado com p_bom_ruim = 0)
    p_bom_ruim: float = 0.0
    p_ruim_bom: float = 0.3
    perda_ruim: float = 0.9
    reordena: float = 0.0
    reordena_s: float = 0.05
    duplica: float = 0.0
    corrompe: float = 0.0
    quedas: list[tuple[float, float]] = field(default_factory=list)
    intermitente: tuple[float, float] | None = None
    fila_s: float = 1.0

    def caido(self, t: float) -> bool:
        if any(a <= t < b for a, b in self.quedas):
            return True
        if self.intermitente:
            ligado, desligado = self.intermitente
            return (t % (ligado + desligado)) >= ligado
        return False


@dataclass
class EstatisticaDoCanal:
    entrou: int = 0
    entregue: int = 0
    perdido: int = 0
    caido: int = 0
    fila_cheia: int = 0
    corrompido: int = 0
    duplicado: int = 0
    bytes: int = 0


class CanalEmulado:
    """Um sentido de um enlace emulado."""

    def __init__(self, modelo: ModeloDeCanal, rng: random.Random) -> None:
        self.modelo = modelo
        self.rng = rng
        self._ruim = False
        self._livre_em = 0.0
        self._fila: list[tuple[float, int, bytes]] = []
        self._n = itertools.count()
        self.est = EstatisticaDoCanal()

    def entra(self, dados: bytes, agora: float) -> None:
        m, rng, est = self.modelo, self.rng, self.est
        est.entrou += 1
        if m.caido(agora):
            est.caido += 1
            return
        # serialização na banda, com fila limitada
        inicio = max(agora, self._livre_em)
        if inicio - agora > m.fila_s:
            est.fila_cheia += 1
            return
        self._livre_em = inicio + len(dados) * 8 / m.banda_bps
        # perdas
        if m.p_bom_ruim > 0:
            if self._ruim and rng.random() < m.p_ruim_bom:
                self._ruim = False
            elif not self._ruim and rng.random() < m.p_bom_ruim:
                self._ruim = True
            if self._ruim and rng.random() < m.perda_ruim:
                est.perdido += 1
                return
        if rng.random() < m.perda:
            est.perdido += 1
            return
        if rng.random() < m.corrompe:
            b = bytearray(dados)
            for _ in range(rng.randint(1, 3)):
                b[rng.randrange(len(b))] ^= rng.randint(1, 255)
            dados = bytes(b)
            est.corrompido += 1
        chegada = self._livre_em + m.atraso_s + rng.uniform(0, m.variacao_s)
        if rng.random() < m.reordena:
            chegada += rng.uniform(0, m.reordena_s)
        heapq.heappush(self._fila, (chegada, next(self._n), dados))
        if rng.random() < m.duplica:
            est.duplicado += 1
            heapq.heappush(self._fila, (chegada + rng.uniform(0, m.reordena_s), next(self._n),
                                        dados))

    def sai(self, agora: float) -> list[bytes]:
        saida = []
        while self._fila and self._fila[0][0] <= agora:
            _, _, dados = heapq.heappop(self._fila)
            # a queda também derruba o que estava "no ar"
            if self.modelo.caido(agora):
                self.est.caido += 1
                continue
            saida.append(dados)
            self.est.entregue += 1
            self.est.bytes += len(dados)
        return saida

    def proxima_chegada(self) -> float | None:
        return self._fila[0][0] if self._fila else None


class RedeEmulada:
    """Pontos em processo, ligados dois a dois por canais emulados (um por sentido)."""

    def __init__(self, relogio, semente: int = 1) -> None:
        self.relogio = relogio
        self._rng = random.Random(semente)
        self._canais: dict[tuple[str, str], CanalEmulado] = {}
        self._pontos: dict[str, PontoEmulado] = {}

    def ponto(self, endereco: str, nome: str, mtu: int = 1200) -> PontoEmulado:
        p = PontoEmulado(self, endereco, nome, mtu)
        self._pontos[endereco] = p
        return p

    def liga(self, a: str, b: str, ida: ModeloDeCanal, volta: ModeloDeCanal | None = None
             ) -> tuple[CanalEmulado, CanalEmulado]:
        c1 = CanalEmulado(ida, random.Random(self._rng.random()))
        c2 = CanalEmulado(volta if volta is not None else replace(ida),
                          random.Random(self._rng.random()))
        self._canais[(a, b)], self._canais[(b, a)] = c1, c2
        return c1, c2

    def canal(self, de: str, para: str) -> CanalEmulado:
        return self._canais[(de, para)]

    def proxima_chegada(self) -> float | None:
        tempos = [t for c in self._canais.values() if (t := c.proxima_chegada()) is not None]
        return min(tempos) if tempos else None


class PontoEmulado:
    def __init__(self, rede: RedeEmulada, endereco: str, nome: str, mtu: int) -> None:
        self.rede = rede
        self.endereco = endereco
        self.nome = nome
        self.mtu = mtu
        self.fechado = False

    def enviar(self, dados: bytes, destino: Any) -> None:
        if self.fechado:
            return
        if len(dados) > self.mtu:
            raise ValueError(f"pacote de {len(dados)} B maior que a MTU {self.mtu}")
        canal = self.rede._canais.get((self.endereco, destino))
        if canal is not None:
            canal.entra(dados, self.rede.relogio.agora())

    def receber(self) -> list[tuple[bytes, Any]]:
        if self.fechado:
            return []
        agora = self.rede.relogio.agora()
        saida = []
        for (de, para), canal in self.rede._canais.items():
            if para == self.endereco:
                saida.extend((d, de) for d in canal.sai(agora))
        return saida

    def fileno(self) -> int | None:
        return None

    def fechar(self) -> None:
        self.fechado = True

    def reabrir(self) -> None:
        """Depois de "matar" o nó dono deste ponto, o processo novo reabre o mesmo endereço."""
        self.fechado = False


# ---------------------------------------------------------------------------
# Relé UDP (tempo real, entre processos)
# ---------------------------------------------------------------------------


class ReleUdp:
    """Relé UDP com o modelo de canal. ``A`` manda para ``escuta``; o relé entrega em
    ``destino`` (``B``) e devolve as respostas de ``B`` ao último endereço visto de ``A``.

    O tempo do modelo (quedas, intermitente) conta a partir da criação do relé.
    """

    def __init__(self, escuta: tuple[str, int], destino: tuple[str, int],
                 ida: ModeloDeCanal, volta: ModeloDeCanal | None = None, *,
                 semente: int = 1) -> None:
        rng = random.Random(semente)
        self.ida = CanalEmulado(ida, random.Random(rng.random()))
        self.volta = CanalEmulado(volta if volta is not None else replace(ida),
                                  random.Random(rng.random()))
        self.destino = destino
        self._lado_a = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._lado_a.bind(escuta)
        self._lado_b = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._lado_b.bind((escuta[0], 0))
        for s in (self._lado_a, self._lado_b):
            s.settimeout(0.002)
        self._a: tuple[str, int] | None = None
        self._t0 = time.monotonic()
        self._parar = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def endereco(self) -> tuple[str, int]:
        return self._lado_a.getsockname()[:2]

    def _agora(self) -> float:
        return time.monotonic() - self._t0

    def _laco(self) -> None:
        while not self._parar.is_set():
            for sock, canal, lado in ((self._lado_a, self.ida, "a"),
                                      (self._lado_b, self.volta, "b")):
                try:
                    while True:
                        dados, origem = sock.recvfrom(65535)
                        if lado == "a":
                            self._a = origem[:2]
                        canal.entra(dados, self._agora())
                except (TimeoutError, BlockingIOError, ConnectionResetError, OSError):
                    pass
            agora = self._agora()
            for dados in self.ida.sai(agora):
                self._lado_b.sendto(dados, self.destino)
            if self._a is not None:
                for dados in self.volta.sai(agora):
                    self._lado_a.sendto(dados, self._a)

    def iniciar(self) -> ReleUdp:
        self._thread = threading.Thread(target=self._laco, daemon=True, name="rele-udp")
        self._thread.start()
        return self

    def parar(self) -> None:
        self._parar.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._lado_a.close()
        self._lado_b.close()


def _modelo_da_linha(args) -> ModeloDeCanal:
    quedas = []
    for q in args.queda or []:
        a, _, b = q.partition("-")
        quedas.append((float(a), float(b)))
    return ModeloDeCanal(
        banda_bps=args.banda, atraso_s=args.atraso_ms / 1000, variacao_s=args.variacao_ms / 1000,
        perda=args.perda, p_bom_ruim=args.rajada, reordena=args.reordena, duplica=args.duplica,
        corrompe=args.corrompe, quedas=quedas,
        intermitente=tuple(map(float, args.intermitente.split(":"))) if args.intermitente
        else None)


def main(argv: list[str] | None = None) -> int:
    from valtexa_rover.enlace.transporte import endereco

    p = argparse.ArgumentParser(prog="python -m valtexa_rover.enlace.emulador",
                                description="Relé UDP com rádio emulado (SIMULAÇÃO).")
    p.add_argument("--escuta", required=True, help="host:porta onde o lado A manda")
    p.add_argument("--destino", required=True, help="host:porta do lado B")
    p.add_argument("--banda", type=float, default=1e6, help="bit/s")
    p.add_argument("--atraso-ms", type=float, default=20)
    p.add_argument("--variacao-ms", type=float, default=0)
    p.add_argument("--perda", type=float, default=0)
    p.add_argument("--rajada", type=float, default=0, help="P(bom→ruim) do Gilbert-Elliott")
    p.add_argument("--reordena", type=float, default=0)
    p.add_argument("--duplica", type=float, default=0)
    p.add_argument("--corrompe", type=float, default=0)
    p.add_argument("--queda", action="append", help="início-fim em segundos (repete)")
    p.add_argument("--intermitente", help="ligado:desligado em segundos")
    p.add_argument("--semente", type=int, default=1)
    args = p.parse_args(argv)
    rele = ReleUdp(endereco(args.escuta), endereco(args.destino), _modelo_da_linha(args),
                   semente=args.semente).iniciar()
    print(f"relé emulado (SIMULAÇÃO) em {rele.endereco} → {rele.destino}", flush=True)
    try:
        while True:
            time.sleep(5)
            print(f"ida {rele.ida.est} | volta {rele.volta.est}", flush=True)
    except KeyboardInterrupt:
        rele.parar()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
