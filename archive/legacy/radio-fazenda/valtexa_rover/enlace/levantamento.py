"""Levantamento de cobertura (Parte 13): a qualidade de cada enlace com a posição RTK.

**Registro** (:class:`RegistradorDeCobertura`): a cada ``intervalo_s`` o Rover — e a base,
com a posição que o Rover manda no batimento — escreve uma linha JSON por vizinho e por
rádio::

    {"t": "2026-10-09T14:03:12.4+00:00", "lado": "rover", "no": "RT-0001", "par": "base",
     "enlace": "alta", "lat": -21.7, "lon": -45.2, "fix": "fixed", "hdop": 0.6,
     "vivo": true, "estado": "vivo", "rtt_ms": 21.4, "perda": 0.03, "vazao_bps": 182000,
     "banda_bps": 20000000, "rssi_dbm": -71, "snr_db": 28}

``rssi_dbm``/``snr_db`` só quando o rádio expõe (adaptador da base, **não testado em
hardware**); RTT, perda e vazão são medidos pelo próprio protocolo. Sem posição boa
(``fix`` diferente de ``fixed``/``float`` quando ``exige_rtk``) a linha não é escrita — a
grade não mistura metro com dezena de metros. O arquivo gira em ``limite_bytes``.

**Mapa** (:class:`MapaDeCobertura`): o que a análise (``valtexa-base levantamento``) devolve
para o Rover — a grade com a classe de cada célula e as **zonas mortas** — e a consulta
"esta posição está numa zona morta?" e "vou entrar numa daqui a quanto?" que a missão
ciente do enlace (:mod:`.travessia`) usa.

Nada aqui lê os arquivos de terreno da IA: a cobertura é a **medida**.
"""

from __future__ import annotations

import itertools
import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from valtexa_rover._compat import UTC

RAIO_TERRA_M = 6_371_000.0
FIX_BONS = ("fixed", "float")


def posicao_de(gps: Any) -> dict | None:
    """``GpsFix`` (HAL) ou dict → ``{"lat", "lon", "fix", "hdop"}``; ``None`` sem leitura."""
    if gps is None:
        return None
    if isinstance(gps, dict):
        if gps.get("lat") is None or gps.get("lon") is None:
            return None
        return {"lat": float(gps["lat"]), "lon": float(gps["lon"]),
                "fix": str(gps.get("fix", "none")), "hdop": gps.get("hdop"),
                "rumo_deg": gps.get("rumo_deg"), "v_ms": gps.get("v_ms")}
    return {"lat": float(gps.lat), "lon": float(gps.lon), "fix": str(gps.fix),
            "hdop": getattr(gps, "hdop", None), "rumo_deg": getattr(gps, "heading_deg", None),
            "v_ms": getattr(gps, "speed_ms", None)}


class RegistradorDeCobertura:
    def __init__(self, caminho: Path | str, *, lado: str, no: str,
                 saude: Callable[[], dict], posicao: Callable[[], dict | None] | None = None,
                 posicao_do_par: Callable[[str], dict | None] | None = None,
                 pares: Callable[[], list[str]] | None = None,
                 intervalo_s: float = 1.0, exige_rtk: bool = True,
                 limite_bytes: int = 200 * 1024 * 1024,
                 parede: Callable[[], float] | None = None) -> None:
        """No Rover: ``posicao()`` é o GNSS dele. Na base: ``posicao_do_par(rover)`` é a
        posição que o Rover mandou no batimento (``pares()`` = os Rovers)."""
        self.caminho = Path(caminho)
        self.lado = lado
        self.no = no
        self.saude = saude
        self.posicao = posicao
        self.posicao_do_par = posicao_do_par
        self.pares = pares
        self.intervalo_s = intervalo_s
        self.exige_rtk = exige_rtk
        self.limite_bytes = limite_bytes
        self.parede = parede or (lambda: datetime.now(UTC).timestamp())
        self._ultimo = -1e18
        self.linhas = 0
        self.sem_posicao = 0

    def _boa(self, pos: dict | None) -> bool:
        return pos is not None and (not self.exige_rtk or pos.get("fix") in FIX_BONS)

    def talvez(self) -> int:
        """Chame a cada volta do laço: amostra quando passou ``intervalo_s``."""
        agora = self.parede()
        if agora - self._ultimo < self.intervalo_s:
            return 0
        self._ultimo = agora
        return self.amostra(agora)

    def amostra(self, agora: float | None = None) -> int:
        agora = self.parede() if agora is None else agora
        pares = self.saude().get("pares", {})
        alvo = self.pares() if self.pares is not None else list(pares)
        linhas = []
        t = datetime.fromtimestamp(agora, UTC).isoformat()
        minha = self.posicao() if self.posicao is not None else None
        for par in alvo:
            p = pares.get(par)
            if p is None:
                continue
            pos = minha if self.posicao_do_par is None else self.posicao_do_par(par)
            if not self._boa(pos):
                self.sem_posicao += 1
                continue
            for enlace, s in (p.get("enlaces") or {}).items():
                linhas.append({"t": t, "lado": self.lado, "no": self.no, "par": par,
                               "enlace": enlace, "lat": pos["lat"], "lon": pos["lon"],
                               "fix": pos.get("fix"), "hdop": pos.get("hdop"),
                               "vivo": s.get("estado") != "morto", "estado": s.get("estado"),
                               "rtt_ms": s.get("rtt_ms"), "perda": s.get("perda"),
                               "vazao_bps": s.get("vazao_bps"), "banda_bps": s.get("banda_bps"),
                               "rssi_dbm": s.get("rssi_dbm"), "snr_db": s.get("snr_db")})
        if linhas:
            self._escreve(linhas)
        return len(linhas)

    def _escreve(self, linhas: list[dict]) -> None:
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        if self.caminho.exists() and self.caminho.stat().st_size > self.limite_bytes:
            os.replace(self.caminho, self.caminho.with_suffix(self.caminho.suffix + ".1"))
        with self.caminho.open("a", encoding="utf-8") as fh:
            for linha in linhas:
                fh.write(json.dumps(linha, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.linhas += len(linhas)


# ---------------------------------------------------------------------------
# Geometria local (equirretangular em torno de uma origem: erro desprezível em 2-3 km)
# ---------------------------------------------------------------------------


def para_metros(lat: float, lon: float, origem: tuple[float, float]) -> tuple[float, float]:
    lat0, lon0 = origem
    x = math.radians(lon - lon0) * RAIO_TERRA_M * math.cos(math.radians(lat0))
    y = math.radians(lat - lat0) * RAIO_TERRA_M
    return x, y


def para_graus(x: float, y: float, origem: tuple[float, float]) -> tuple[float, float]:
    lat0, lon0 = origem
    lat = lat0 + math.degrees(y / RAIO_TERRA_M)
    lon = lon0 + math.degrees(x / (RAIO_TERRA_M * math.cos(math.radians(lat0))))
    return lat, lon


@dataclass
class ZonaMorta:
    id: int
    celulas: set[tuple[int, int]]
    centro: tuple[float, float]
    area_m2: float


class MapaDeCobertura:
    """A grade da análise, no Rover: classe por célula e zonas mortas."""

    def __init__(self, dados: dict) -> None:
        self.dados = dados
        self.origem = (float(dados["origem"][0]), float(dados["origem"][1]))
        self.celula_m = float(dados["celula_m"])
        self.celulas: dict[tuple[int, int], str] = {
            tuple(int(v) for v in k.split(",")): c for k, c in dados["celulas"].items()}
        self.zonas = [ZonaMorta(z["id"], {tuple(c) for c in z["celulas"]},
                                tuple(z["centro"]), float(z["area_m2"]))
                      for z in dados.get("zonas_mortas", [])]
        self._zona_da_celula = {c: z for z in self.zonas for c in z.celulas}

    @classmethod
    def carregar(cls, caminho: Path | str) -> MapaDeCobertura:
        return cls(json.loads(Path(caminho).read_text(encoding="utf-8")))

    def celula(self, lat: float, lon: float) -> tuple[int, int]:
        x, y = para_metros(lat, lon, self.origem)
        return math.floor(x / self.celula_m), math.floor(y / self.celula_m)

    def classe(self, lat: float, lon: float) -> str | None:
        """``bom``, ``marginal``, ``morto`` ou ``None`` (nunca medido)."""
        return self.celulas.get(self.celula(lat, lon))

    def zona(self, lat: float, lon: float) -> ZonaMorta | None:
        return self._zona_da_celula.get(self.celula(lat, lon))

    def proxima_zona(self, pontos: list[tuple[float, float]], passo_m: float | None = None
                     ) -> tuple[ZonaMorta, float, float] | None:
        """Ao longo do caminho ``pontos`` (lat, lon): a primeira zona morta, a distância até a
        entrada e o comprimento da travessia (m). ``None`` se o caminho não cruza nenhuma."""
        passo = passo_m or self.celula_m / 3
        andado = 0.0
        dentro: ZonaMorta | None = None
        entrada = 0.0
        for (la1, lo1), (la2, lo2) in itertools.pairwise(pontos):
            x1, y1 = para_metros(la1, lo1, self.origem)
            x2, y2 = para_metros(la2, lo2, self.origem)
            seg = math.hypot(x2 - x1, y2 - y1)
            n = max(1, math.ceil(seg / passo))
            for k in range(n):
                f = k / n
                x, y = x1 + (x2 - x1) * f, y1 + (y2 - y1) * f
                c = (math.floor(x / self.celula_m), math.floor(y / self.celula_m))
                z = self._zona_da_celula.get(c)
                d = andado + seg * f
                if dentro is None and z is not None:
                    dentro, entrada = z, d
                elif dentro is not None and z is not dentro:
                    return dentro, entrada, d - entrada
            andado += seg
        if dentro is not None:
            return dentro, entrada, andado - entrada
        return None
