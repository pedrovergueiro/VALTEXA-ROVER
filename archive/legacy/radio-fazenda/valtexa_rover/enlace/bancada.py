"""Bancada **SIMULADA** do enlace: um Rover e uma base em relógio virtual, rádios emulados.

Usada pelos testes de caos e pelo relatório de latência (``python -m
valtexa_rover.enlace.bancada``). Matar e reiniciar um nó = descartar o objeto sem fechar
nada (como um ``kill -9``) e criar outro sobre o mesmo SQLite.
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path

from valtexa_rover.enlace.cripto import ChaveDeEnlace
from valtexa_rover.enlace.emulador import ModeloDeCanal, RedeEmulada
from valtexa_rover.enlace.fila import FilaPersistente, Limites
from valtexa_rover.enlace.no import ConfigDoEnlace, NoDeEnlace, ParametrosDoNo
from valtexa_rover.enlace.transporte import RelogioDeslocado, RelogioVirtual

# Perfis de rádio do emulador. NÃO são medidas: são hipóteses para exercitar o protocolo.
SUBGHZ = ModeloDeCanal(banda_bps=250_000, atraso_s=0.04, variacao_s=0.02, fila_s=2.0)
ALTA = ModeloDeCanal(banda_bps=20_000_000, atraso_s=0.005, variacao_s=0.005, fila_s=0.5)


def percentis(valores: list[float]) -> dict[str, float]:
    if not valores:
        return {"n": 0}
    v = sorted(valores)

    def p(q: float) -> float:
        return v[min(len(v) - 1, round(q * (len(v) - 1)))]

    return {"n": len(v), "p50_ms": round(p(0.50) * 1000, 1), "p95_ms": round(p(0.95) * 1000, 1),
            "p99_ms": round(p(0.99) * 1000, 1), "max_ms": round(v[-1] * 1000, 1),
            "media_ms": round(statistics.fmean(v) * 1000, 1)}


@dataclass
class Bancada:
    pasta: Path
    modelos: dict[str, ModeloDeCanal] = field(default_factory=lambda: {"subghz": SUBGHZ,
                                                                       "alta": ALTA})
    semente: int = 7
    limites: Limites | None = None
    parametros: ParametrosDoNo | None = None
    passo_s: float = 0.01
    desvio_parede_rover_s: float = 0.0
    """Relógio de parede do Rover errado (o monotônico é o mesmo)."""

    def __post_init__(self) -> None:
        # cópia própria de cada modelo: mexer nas quedas de um teste não muda os perfis
        self.modelos = {n: replace(m, quedas=list(m.quedas)) for n, m in self.modelos.items()}
        self.relogio = RelogioVirtual()
        self.relogio_rover = (RelogioDeslocado(self.relogio, self.desvio_parede_rover_s)
                              if self.desvio_parede_rover_s else self.relogio)
        self.rede = RedeEmulada(self.relogio, self.semente)
        self.k_rover = ChaveDeEnlace.gerar()
        self.k_base = ChaveDeEnlace.gerar()
        self.canais = {}
        for nome, modelo in self.modelos.items():
            self.rede.ponto(f"rover:{nome}", nome)
            self.rede.ponto(f"base:{nome}", nome)
            self.canais[nome] = self.rede.liga(f"rover:{nome}", f"base:{nome}", modelo)
        self.rover = self.novo_rover()
        self.base = self.nova_base()

    def _pontos(self, lado: str):
        pts = self.rede._pontos
        for nome in self.modelos:
            pts[f"{lado}:{nome}"].reabrir()
        return [pts[f"{lado}:{nome}"] for nome in self.modelos]

    def novo_rover(self) -> NoDeEnlace:
        enlaces = [ConfigDoEnlace(n, p, self.modelos[n].banda_bps, f"base:{n}")
                   for n, p in zip(self.modelos, self._pontos("rover"), strict=True)]
        return NoDeEnlace("rover", self.k_rover,
                          FilaPersistente(self.pasta / "rover.db", self.relogio_rover.parede,
                                          limites=self.limites, sincrono="NORMAL"),
                          self.relogio_rover, enlaces, {"base": self.k_base.publica},
                          iniciador=True,
                          parametros=self.parametros)

    def nova_base(self, info=None) -> NoDeEnlace:
        enlaces = [ConfigDoEnlace(n, p, self.modelos[n].banda_bps)
                   for n, p in zip(self.modelos, self._pontos("base"), strict=True)]
        return NoDeEnlace("base", self.k_base,
                          FilaPersistente(self.pasta / "base.db", self.relogio.parede,
                                          limites=self.limites, sincrono="NORMAL"),
                          self.relogio, enlaces, {"rover": self.k_rover.publica},
                          iniciador=False, parametros=self.parametros, info_batimento=info)

    def mata_rover(self) -> None:
        """``kill -9``: nada é fechado nem gravado além do que já estava no disco."""
        for e in self.rover.enlaces.values():
            e.transporte.fechar()
        self.rover = self.novo_rover()

    def mata_base(self) -> None:
        for e in self.base.enlaces.values():
            e.transporte.fechar()
        self.base = self.nova_base()

    def anda(self, segundos: float, a_cada=None) -> None:
        fim = self.relogio.agora() + segundos
        while self.relogio.agora() < fim:
            self.rover.passo()
            self.base.passo()
            if a_cada is not None:
                a_cada()
            self.relogio.avanca(self.passo_s)


def relatorio(pasta: Path, segundos: float = 120.0) -> dict:
    """P1 sob carga de P4 com 30 % de perda nos dois enlaces (o número que o doc registra)."""
    perda = {"subghz": ModeloDeCanal(**{**SUBGHZ.__dict__, "perda": 0.3}),
             "alta": ModeloDeCanal(**{**ALTA.__dict__, "perda": 0.3})}
    b = Bancada(pasta, perda)
    enviados: dict[bytes, float] = {}
    lat: list[float] = []
    proximo = [0.0]

    def carga() -> None:
        agora = b.relogio.agora()
        if agora >= proximo[0]:
            proximo[0] = agora + 0.25
            corpo = b"cmd-%f" % agora + b"x" * 300
            enviados[corpo[:20]] = agora
            b.rover.enviar("base", corpo, 1)
        if b.rover.fila.tamanho_saida().get(4, 0) < 8:
            b.rover.enviar("base", b"o" * 200_000, 4)
        for c in b.base.chegadas(None, 500):
            b.base.consumida(c.seq)
            if c.prio == 1:
                lat.append(agora - enviados.pop(c.corpo[:20]))

    b.anda(segundos, carga)
    return {"p1": percentis(lat), "p1_perdidos": len(enviados),
            "saude_rover": b.rover.saude()["pares"]["base"]["enlaces"]}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m valtexa_rover.enlace.bancada",
                                description="Latência de P1 sob carga, rádio EMULADO.")
    p.add_argument("--segundos", type=float, default=120)
    args = p.parse_args(argv)
    with tempfile.TemporaryDirectory() as d:
        print(json.dumps(relatorio(Path(d), args.segundos), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
