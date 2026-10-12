"""Perfil **SIMULADO** de cafezal em morro, para os testes e o ensaio de 24 h acelerado.

**Não é medida.** É um modelo de brinquedo, explicável, para exercitar o protocolo, a malha,
o levantamento e a missão ciente do enlace com o tipo de coisa que o Sul de Minas tem: ruas
de 3,5 m, copa de 2 a 3 m que atenua o rádio de forma periódica ao longo da rua, e um morro
entre a base e o fundo do talhão que faz **sombra** (zona morta para o rádio de alta vazão;
o sub-GHz difrata e fica só marginal). Os números certos vêm do levantamento de campo.

Coordenadas locais em metros: ``x`` atravessa as ruas (leste), ``y`` corre ao longo delas
(norte). A base fica a oeste do talhão.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, replace

from valtexa_rover.enlace.emulador import ModeloDeCanal
from valtexa_rover.enlace.levantamento import para_graus

ORIGEM = (-21.7500, -45.2500)  # Sul de Minas aproximado (só para ter lat/lon plausível)


@dataclass(frozen=True)
class Talhao:
    largura_m: float = 240.0
    comprimento_m: float = 200.0
    rua_m: float = 3.5
    base_xy: tuple[float, float] = (-40.0, 100.0)
    # sombra do morro: o fundo-leste do talhão
    sombra_x: tuple[float, float] = (150.0, 240.0)
    sombra_y: tuple[float, float] = (0.0, 90.0)
    repetidora_xy: tuple[float, float] | None = None
    """Repetidora no topo do morro (cobre a sombra), se instalada."""

    def na_sombra(self, x: float, y: float) -> bool:
        return self.sombra_x[0] <= x <= self.sombra_x[1] and \
            self.sombra_y[0] <= y <= self.sombra_y[1]

    def latlon(self, x: float, y: float) -> tuple[float, float]:
        return para_graus(x, y, ORIGEM)

    @property
    def base_latlon(self) -> tuple[float, float]:
        return self.latlon(*self.base_xy)


def caminho(t: Talhao, v_ms: float = 0.8, passo_s: float = 1.0,
            ruas: range | None = None) -> Iterator[tuple[float, float, float]]:
    """``(segundos, x, y)``: o Rover percorre as ruas em zigue-zague (ida e volta)."""
    n_ruas = int(t.largura_m // t.rua_m)
    tempo = 0.0
    for k in ruas if ruas is not None else range(n_ruas):
        x = (k + 0.5) * t.rua_m
        ida = k % 2 == 0
        y, fim = (0.0, t.comprimento_m) if ida else (t.comprimento_m, 0.0)
        sentido = 1 if ida else -1
        while (y - fim) * sentido < 0:
            yield tempo, x, y
            y += sentido * v_ms * passo_s
            tempo += passo_s
        # manobra na cabeceira (ainda dentro do talhão: conta)
        for _ in range(int(t.rua_m / v_ms / passo_s) + 1):
            yield tempo, x, fim
            tempo += passo_s


def rssi(t: Talhao, enlace: str, x: float, y: float, de: tuple[float, float] | None = None
         ) -> float:
    """RSSI (dBm) modelado. Espaço livre + copa periódica + sombra do morro (SIMULADO)."""
    bx, by = de or t.base_xy
    d = max(5.0, math.hypot(x - bx, y - by))
    copa = 6.0 * (0.5 + 0.5 * math.sin(2 * math.pi * y / 7.0))  # entre plantas e na planta
    if enlace == "subghz":
        valor = -32.0 - 20 * math.log10(d) - 0.3 * copa
        if de is None and t.na_sombra(x, y):
            valor -= 22.0  # difrata mal no morro: a sombra some também
        return valor
    valor = -38.0 - 20 * math.log10(d) - copa
    if de is None and t.na_sombra(x, y):
        valor -= 35.0  # sem visada: some
    return valor


LIMIAR_DBM = {"subghz": -100.0, "alta": -88.0}


def qualidade(t: Talhao, enlace: str, x: float, y: float,
              de: tuple[float, float] | None = None) -> dict:
    """``{"vivo", "perda", "rtt_ms", "rssi_dbm", "snr_db"}`` no ponto (SIMULADO)."""
    r = rssi(t, enlace, x, y, de)
    margem = r - LIMIAR_DBM[enlace]
    vivo = margem > 0
    perda = 1.0 if not vivo else min(0.6, max(0.0, 0.5 * math.exp(-margem / 4.0)))
    rtt = None if not vivo else (40.0 if enlace == "subghz" else 6.0) * (1 + 2 * perda)
    return {"vivo": vivo, "perda": round(perda, 3), "rtt_ms": rtt, "rssi_dbm": round(r, 1),
            "snr_db": round(r + 95.0, 1)}


def modelo(base: ModeloDeCanal, q: dict) -> ModeloDeCanal:
    """O canal emulado com a perda do ponto (morto = perda total)."""
    return replace(base, perda=1.0 if not q["vivo"] else q["perda"])


def amostras_de_levantamento(t: Talhao, *, v_ms: float = 0.8, passo_s: float = 1.0,
                             ruas: range | None = None, no: str = "RT-0001") -> list[dict]:
    """O que o registrador do Rover gravaria percorrendo o talhão (para testar a análise)."""
    from datetime import UTC, datetime, timedelta

    t0 = datetime(2026, 10, 9, 8, 0, tzinfo=UTC)
    saida = []
    for s, x, y in caminho(t, v_ms, passo_s, ruas):
        lat, lon = t.latlon(x, y)
        quando = (t0 + timedelta(seconds=s)).isoformat()
        vizinhos = [("base", None)]
        if t.repetidora_xy is not None:
            vizinhos.append(("rep-morro", t.repetidora_xy))
        for par, de in vizinhos:
            for enlace in ("subghz", "alta"):
                q = qualidade(t, enlace, x, y, de)
                saida.append({"t": quando, "lado": "rover", "no": no, "par": par,
                              "enlace": enlace, "lat": lat, "lon": lon, "fix": "fixed",
                              "hdop": 0.6, "vivo": q["vivo"],
                              "estado": "vivo" if q["vivo"] else "morto",
                              "rtt_ms": q["rtt_ms"], "perda": q["perda"],
                              "vazao_bps": None, "banda_bps": 2.5e5 if enlace == "subghz"
                              else 2e7, "rssi_dbm": q["rssi_dbm"], "snr_db": q["snr_db"]})
    return saida


def mapa_simulado(t: Talhao, celula_m: float = 10.0) -> dict:
    """O ``cobertura.json`` que um levantamento perfeito deste talhão daria (SIMULADO).

    Célula ``morto`` quando nenhum rádio fecha no centro dela; ``marginal`` quando só o
    sub-GHz fecha. Para testar a missão ciente do enlace sem depender da análise (que tem
    teste próprio em ``base/tests/test_levantamento.py``).
    """
    from valtexa_rover.enlace.levantamento import para_metros

    origem = t.base_latlon
    bx, by = para_metros(*t.latlon(0.0, 0.0), origem)
    celulas: dict[str, str] = {}
    mortas: set[tuple[int, int]] = set()
    i0, i1 = math.floor(bx / celula_m), math.floor((bx + t.largura_m) / celula_m)
    j0, j1 = math.floor(by / celula_m), math.floor((by + t.comprimento_m) / celula_m)
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            x = (i + 0.5) * celula_m - bx
            y = (j + 0.5) * celula_m - by
            vivos = [e for e in ("subghz", "alta") if qualidade(t, e, x, y)["vivo"]]
            classe = "bom" if "alta" in vivos else "marginal" if vivos else "morto"
            celulas[f"{i},{j}"] = classe
            if classe == "morto":
                mortas.add((i, j))
    zonas = []
    if mortas:
        xs = [(i + 0.5) * celula_m for i, _ in mortas]
        ys = [(j + 0.5) * celula_m for _, j in mortas]
        lat, lon = para_graus(sum(xs) / len(xs), sum(ys) / len(ys), origem)
        zonas.append({"id": 1, "celulas": sorted([list(c) for c in mortas]),
                      "centro": [lat, lon], "area_m2": len(mortas) * celula_m ** 2})
    return {"versao": 1, "origem": list(origem), "celula_m": celula_m, "celulas": celulas,
            "zonas_mortas": zonas}
