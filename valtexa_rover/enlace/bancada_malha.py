"""Bancada **SIMULADA** da malha: Rover, repetidoras e base em relógio virtual, rádios emulados.

Topologia livre: ``nos`` (nome → papel) e ``ligacoes`` (quem ouve quem, em que rádio, com que
canal). Cada nó tem uma interface por rádio (``"<nome>:<rádio>"``); o mesmo rádio de um nó
pode ouvir vários vizinhos (meio compartilhado: o balde de banda é da interface).

Quem inicia o aperto de mão com quem: o nó mais "longe" da base inicia com o mais perto
(``nivel``: Rover 0, repetidora 1..n, base 99; empate pelo nome). Matar um nó = parar de
rodar e fechar as interfaces sem gravar nada (``kill -9``); reviver = processo novo sobre
o mesmo SQLite. Nenhum número daqui é medida de rádio.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field, replace
from pathlib import Path

from valtexa_rover.enlace.cripto import ChaveDeEnlace
from valtexa_rover.enlace.emulador import ModeloDeCanal, RedeEmulada
from valtexa_rover.enlace.fila import FilaPersistente, Limites
from valtexa_rover.enlace.malha import Malha, ParametrosDaMalha
from valtexa_rover.enlace.no import ConfigDoEnlace, NoDeEnlace, ParametrosDoNo
from valtexa_rover.enlace.transporte import RelogioVirtual

NIVEL = {"rover": 0, "repetidora": 1, "base": 99}


@dataclass
class Ligacao:
    a: str
    b: str
    radio: str
    modelo: ModeloDeCanal


@dataclass
class NoDaBancada:
    nome: str
    papel: str
    nivel: int
    chave: ChaveDeEnlace
    radios: dict[str, float] = field(default_factory=dict)  # rádio -> banda da interface
    vizinhos: dict[str, list[str]] = field(default_factory=dict)  # vizinho -> rádios
    malha: Malha | None = None
    vivo: bool = True
    atributos: dict = field(default_factory=dict)
    custo_extra_s: float = 0.0


class BancadaMalha:
    def __init__(self, pasta: Path, nos: dict[str, str], ligacoes: list[Ligacao], *,
                 semente: int = 11, passo_s: float = 0.01, niveis: dict[str, int] | None = None,
                 parametros: ParametrosDoNo | None = None,
                 parametros_malha: ParametrosDaMalha | None = None,
                 limites: Limites | None = None) -> None:
        self.pasta = Path(pasta)
        self.passo_s = passo_s
        self.relogio = RelogioVirtual()
        self.rede = RedeEmulada(self.relogio, semente)
        self.parametros = parametros
        self.parametros_malha = parametros_malha
        self.limites = limites
        self.nos: dict[str, NoDaBancada] = {}
        for nome, papel in nos.items():
            nivel = (niveis or {}).get(nome, NIVEL[papel])
            self.nos[nome] = NoDaBancada(nome, papel, nivel, ChaveDeEnlace.gerar())
        self.ligacoes = [replace(lg, modelo=replace(lg.modelo, quedas=list(lg.modelo.quedas)))
                         for lg in ligacoes]
        self.canais = {}
        for lg in self.ligacoes:
            for x, y in ((lg.a, lg.b), (lg.b, lg.a)):
                n = self.nos[x]
                n.radios[lg.radio] = max(n.radios.get(lg.radio, 0.0), lg.modelo.banda_bps)
                n.vizinhos.setdefault(y, []).append(lg.radio)
        for n in self.nos.values():
            for radio in n.radios:
                self.rede.ponto(f"{n.nome}:{radio}", radio)
        for lg in self.ligacoes:
            self.canais[(lg.a, lg.b, lg.radio)] = self.rede.liga(
                f"{lg.a}:{lg.radio}", f"{lg.b}:{lg.radio}", lg.modelo)
        self.pontas = [n.nome for n in self.nos.values() if n.papel in ("rover", "base")]
        for n in self.nos.values():
            self._sobe(n)

    # -- nós --------------------------------------------------------------------------

    def _inicia_com(self, n: NoDaBancada, viz: str) -> bool:
        v = self.nos[viz]
        return (n.nivel, n.nome) < (v.nivel, v.nome)

    def _sobe(self, n: NoDaBancada) -> None:
        pontos = self.rede._pontos
        enlaces = []
        for radio, banda in n.radios.items():
            ponto = pontos[f"{n.nome}:{radio}"]
            ponto.reabrir()
            enlaces.append(ConfigDoEnlace(radio, ponto, banda))
        pares = {viz: self.nos[viz].chave.publica for viz in n.vizinhos}
        destinos = {viz: {r: f"{viz}:{r}" for r in radios}
                    for viz, radios in n.vizinhos.items() if self._inicia_com(n, viz)}
        no = NoDeEnlace(n.nome, n.chave,
                        FilaPersistente(self.pasta / f"{n.nome}.db", self.relogio.parede,
                                        limites=self.limites, sincrono="NORMAL"),
                        self.relogio, enlaces, pares, iniciador=False,
                        parametros=self.parametros, destinos=destinos)
        publicas = {o: self.nos[o].chave.publica for o in self.pontas if o != n.nome} \
            if n.papel in ("rover", "base") else {}
        n.malha = Malha(n.nome, no, self.pasta / f"{n.nome}-malha.db",
                        parametros=self.parametros_malha, atributos=lambda n=n: dict(n.atributos),
                        custo_extra=lambda n=n: n.custo_extra_s, publicas=publicas,
                        sincrono="NORMAL")
        n.vivo = True

    def malha(self, nome: str) -> Malha:
        m = self.nos[nome].malha
        assert m is not None
        return m

    def mata(self, nome: str) -> None:
        """``kill -9`` (ou bateria acabou): interfaces fechadas, nada gravado além do disco."""
        n = self.nos[nome]
        for e in n.malha.no.enlaces.values():
            e.transporte.fechar()
        n.vivo = False

    def revive(self, nome: str) -> None:
        self._sobe(self.nos[nome])

    def corta(self, a: str, b: str, radio: str, de: float, ate: float) -> None:
        """Queda programada da ligação ``a``-``b`` naquele rádio (nos dois sentidos)."""
        for c in self.canais[(a, b, radio)] if (a, b, radio) in self.canais else \
                self.canais[(b, a, radio)]:
            c.modelo.quedas.append((de, ate))

    # -- tempo ------------------------------------------------------------------------

    def anda(self, segundos: float, a_cada=None) -> None:
        fim = self.relogio.agora() + segundos
        while self.relogio.agora() < fim:
            for n in self.nos.values():
                if n.vivo:
                    n.malha.passo()
            if a_cada is not None:
                a_cada()
            self.relogio.avanca(self.passo_s)

    def ate(self, condicao, limite_s: float, a_cada=None, passo_s: float = 0.1) -> float:
        """Anda até ``condicao()``; devolve quanto levou (``inf`` se não aconteceu)."""
        t0 = self.relogio.agora()
        while self.relogio.agora() - t0 < limite_s:
            if condicao():
                return self.relogio.agora() - t0
            self.anda(passo_s, a_cada)
        return float("inf") if not condicao() else self.relogio.agora() - t0


class Placar:
    """Quem mandou o quê pela malha, quando; quem recebeu o quê, quando e quantas vezes."""

    def __init__(self, b: BancadaMalha) -> None:
        self.b = b
        self.enviadas: dict[bytes, tuple[str, str, int, float]] = {}
        self.recebidas: dict[bytes, list[float]] = {}

    def manda(self, de: str, para: str, prio: int, tamanho: int = 200,
              ttl_s: float | None = None) -> bytes:
        n = len(self.enviadas)
        cabeca = f"{de}>{para}:{prio}:{n:06d}|".encode()
        corpo = cabeca + b"." * max(0, tamanho - len(cabeca))
        self.b.malha(de).enviar(para, corpo, prio, ttl_s=ttl_s)
        self.enviadas[cabeca] = (de, para, prio, self.b.relogio.agora())
        return cabeca

    def colhe(self) -> None:
        agora = self.b.relogio.agora()
        for n in self.b.nos.values():
            if not n.vivo:
                continue
            for c in n.malha.chegadas(None, 1000):
                n.malha.consumida(c.seq)
                if b"|" not in c.corpo:
                    continue
                cabeca = c.corpo[:c.corpo.index(b"|") + 1]
                self.recebidas.setdefault(cabeca, []).append(agora)

    def faltam(self, prios=(0, 1, 2, 3)) -> list[bytes]:
        return [k for k, (_, _, p, _) in self.enviadas.items() if p in prios
                and k not in self.recebidas]

    def repetidas(self) -> list[bytes]:
        return [k for k, v in self.recebidas.items() if len(v) > 1]

    def latencias(self, prio: int) -> list[float]:
        return [self.recebidas[k][0] - t for k, (_, _, p, t) in self.enviadas.items()
                if p == prio and k in self.recebidas]


def resumo(valores: list[float]) -> dict:
    if not valores:
        return {"n": 0}
    v = sorted(valores)

    def p(q: float) -> float:
        return v[min(len(v) - 1, round(q * (len(v) - 1)))]

    return {"n": len(v), "p50_ms": round(p(0.5) * 1000, 1), "p95_ms": round(p(0.95) * 1000, 1),
            "p99_ms": round(p(0.99) * 1000, 1), "max_ms": round(v[-1] * 1000, 1),
            "media_ms": round(statistics.fmean(v) * 1000, 1)}
