"""Contas dos ensaios: percentis, precisão de GNSS (CEP/2DRMS), inclinação pela gravidade.

Funções puras, sem hardware — as mesmas contas são refeitas na plataforma
(``saas/valtexa/services/ensaios.py``) a partir das amostras brutas que o robô envia: o
veredito que vale é o do servidor.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

GRAVIDADE_PADRAO = 9.80665
"""m/s² (gravidade padrão). No Sul de Minas a local é ~9,784 — diferença de 0,2 %, bem
abaixo da tolerância do ensaio; a IMU parada deve medir algo entre as duas."""

RAIO_TERRA_M = 6_378_137.0


def media(valores: Sequence[float]) -> float:
    if not valores:
        raise ValueError("média de lista vazia")
    return sum(valores) / len(valores)


def desvio(valores: Sequence[float]) -> float:
    """Desvio-padrão amostral (n - 1); 0 com menos de duas amostras."""
    if len(valores) < 2:
        return 0.0
    m = media(valores)
    return math.sqrt(sum((v - m) ** 2 for v in valores) / (len(valores) - 1))


def percentil(valores: Sequence[float], p: float) -> float:
    """Percentil com interpolação linear (o ``numpy.percentile`` padrão)."""
    if not valores:
        raise ValueError("percentil de lista vazia")
    ordenados = sorted(valores)
    if len(ordenados) == 1:
        return float(ordenados[0])
    pos = (len(ordenados) - 1) * p / 100.0
    baixo = math.floor(pos)
    alto = min(baixo + 1, len(ordenados) - 1)
    return float(ordenados[baixo] + (ordenados[alto] - ordenados[baixo]) * (pos - baixo))


def mediana(valores: Sequence[float]) -> float:
    return percentil(valores, 50.0)


def enu(lat: float, lon: float, alt: float | None, lat0: float, lon0: float,
        alt0: float | None) -> tuple[float, float, float]:
    """Leste, norte e cima (m) de um ponto em relação à origem — plano tangente local.

    Aproximação equiretangular: em distâncias de dezenas de metros o erro é de micrômetros,
    muito abaixo do centímetro do RTK.
    """
    e = math.radians(lon - lon0) * RAIO_TERRA_M * math.cos(math.radians(lat0))
    n = math.radians(lat - lat0) * RAIO_TERRA_M
    u = (alt - alt0) if alt is not None and alt0 is not None else 0.0
    return e, n, u


def precisao_horizontal(lats: Sequence[float], lons: Sequence[float],
                        ref: tuple[float, float] | None = None) -> dict[str, float]:
    """CEP50, CEP95 e 2DRMS (cm) das posições, em torno do ponto conhecido ou da média.

    * CEP50/CEP95: raio que contém 50 %/95 % das posições (percentil do erro radial);
    * 2DRMS: ``2·√(média(e² + n²))`` — contém ~95-98 % das posições;
    * com ponto conhecido, também o viés (distância da média ao ponto): é o que o RTK
      erra de forma sistemática (base mal levantada, antena fora do lugar).
    """
    if len(lats) != len(lons) or not lats:
        raise ValueError("posições vazias ou desencontradas")
    lat0, lon0 = ref if ref is not None else (media(lats), media(lons))
    pontos = [enu(la, lo, None, lat0, lon0, None)[:2] for la, lo in zip(lats, lons, strict=True)]
    raios = [math.hypot(e, n) for e, n in pontos]
    saida = {
        "cep50_cm": 100 * percentil(raios, 50),
        "cep95_cm": 100 * percentil(raios, 95),
        "drms2_cm": 100 * 2 * math.sqrt(media([e * e + n * n for e, n in pontos])),
        "amostras": float(len(raios)),
    }
    me, mn = media([p[0] for p in pontos]), media([p[1] for p in pontos])
    saida["vies_cm"] = 100 * math.hypot(me, mn) if ref is not None else 0.0
    return saida


def inclinacao_pela_gravidade(ax: float, ay: float, az: float) -> tuple[float, float]:
    """Rolagem e arfagem (graus) que a gravidade medida implica, robô parado.

    Convenção da Xsens (x para a frente, y para a esquerda, z para cima; o acelerômetro
    parado lê +g em z): ``roll = atan2(ay, az)``, ``pitch = atan2(-ax, √(ay² + az²))``.
    Se os eixos do sensor estiverem trocados em relação ao robô, estes ângulos discordam
    dos que o próprio sensor publica — é o que o ensaio de IMU procura.
    """
    roll = math.degrees(math.atan2(ay, az))
    pitch = math.degrees(math.atan2(-ax, math.hypot(ay, az)))
    return roll, pitch


def diferenca_angular(a: float, b: float) -> float:
    """``a - b`` em graus, no intervalo (-180, 180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


def taxas(angulos: Sequence[float], tempos: Sequence[float]) -> list[float]:
    """Derivada (graus/s) de uma série de ângulos, desembrulhando a volta de 360°."""
    saida = []
    for i in range(1, len(angulos)):
        dt = tempos[i] - tempos[i - 1]
        if dt > 0:
            saida.append(diferenca_angular(angulos[i], angulos[i - 1]) / dt)
    return saida
