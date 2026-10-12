"""Geometria de captura do rover: de que parte da planta vem cada recorte.

O rover v3 (``VALTEXA-SAAS/hardware/rover-v3``, rev E) fotografa a planta **de
lado**, não folha a folha: seis OAK-D Pro W PoE em três andares de um mastro
(Z = 450, 1.000 e 1.540 mm), perpendiculares à fileira, com sensor de 12 MP. A
cada quadro entra um pedaço de ~1 m de planta — dezenas de folhas, o pseudocaule
inteiro da bananeira, o tronco do citros.

Os modelos, por outro lado, foram treinados com **uma folha, ou um trecho de
caule, preenchendo o quadro**: 15 a 25 cm de planta em 224 px. Reduzir o quadro
inteiro de 4.056 px para 224 transformaria uma lesão de Sigatoka em um pixel.

Então o quadro é cortado em recortes de **tamanho físico fixo** antes de ir ao
modelo. O tamanho em pixels sai da distância: a OAK-D é RGB-D, e a profundidade
diz a quantos milímetros a planta está — a 270 mm (citros, café) um recorte de
150 mm tem ~580 px; a 1 m (pseudocaule da bananeira, entre linhas mais largas),
~160 px. E como a altura da câmera é conhecida, cada recorte sabe a que altura
do chão está: é o que separa "pseudocaule" de "folha baixa".

## O que é fonte e o que é hipótese

- alturas das câmeras, 12 MP e FOV horizontal de ~127°: projeto rev E, FONTE;
- FOV vertical de 110°: **alvo de projeto, A VALIDAR** contra o datasheet;
- modelo de câmera: pinhole. A lente grande-angular distorce nas bordas, e a
  altura estimada perde precisão lá — sem a calibração intrínseca da câmera,
  não há como corrigir;
- faixas de altura por cultura (:data:`ZONES`): **proposta**, a validar em campo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

SIDE_CAMERA_HEIGHTS_MM = (450.0, 1000.0, 1540.0)
"""Andares das câmeras laterais no mastro (rev E, ``CAMERAS_Z``)."""

HFOV_DEG = 127.0
"""FOV horizontal publicado do OAK-D Pro W."""

VFOV_DEG = 110.0
"""FOV vertical de projeto — A VALIDAR com o fabricante."""

FRAME_PX = (4056, 3040)
"""Quadro de 12 MP do sensor RGB, largura x altura."""

DEFAULT_TILE_MM = 150.0
"""Lado físico do recorte: uma folha de citros inteira, ou um trecho de pseudocaule.

É a escala das fotos de treino — folha preenchendo o quadro. Recorte maior faz a
lesão sumir; menor corta a folha e tira o contexto que o modelo usa.
"""

MIN_TILE_PX = 112
"""Abaixo disso o recorte é ampliado mais de 2x para chegar a 224 e só inventa pixel."""

ZONES: dict[str, tuple[tuple[str, float, float], ...]] = {
    "banana": (
        ("pseudocaule", 0.0, 1200.0),
        ("folhas_baixas", 900.0, 2500.0),
    ),
    "citrus": (
        ("tronco_e_saia", 0.0, 500.0),
        ("copa", 350.0, 2500.0),
    ),
    # Os terços do cafeeiro, como o agrônomo amostra (a ferrugem se avalia no terço
    # médio; o bicho-mineiro, no terço superior). Pé adulto de ~2,2 m.
    "coffee": (
        ("terco_inferior", 0.0, 800.0),
        ("terco_medio", 700.0, 1500.0),
        ("terco_superior", 1400.0, 2600.0),
    ),
}
"""Faixas de altura (mm do chão) por cultura. Proposta — as sobreposições são de
propósito: um recorte na fronteira pertence às duas zonas."""


@dataclass(frozen=True)
class Tile:
    """Um recorte do quadro, com onde ele está na planta."""

    box: tuple[int, int, int, int]
    """``(x0, y0, x1, y1)`` em pixels do quadro."""

    center_height_mm: float
    """Altura estimada do centro do recorte em relação ao chão."""

    zones: tuple[str, ...]
    """Zonas da cultura que o recorte toca, ex.: ``("pseudocaule",)``."""


def gsd_mm_per_px(distance_mm: float, fov_deg: float, pixels: int) -> float:
    """Milímetros de planta por pixel, a uma distância, num eixo da imagem."""
    if distance_mm <= 0 or pixels <= 0 or not 0 < fov_deg < 180:
        raise ValueError(f"Geometria inválida: {distance_mm} mm, {fov_deg}°, {pixels} px.")
    return 2.0 * distance_mm * math.tan(math.radians(fov_deg) / 2.0) / pixels


def plan_tiles(
    frame_size: tuple[int, int],
    distance_mm: float,
    camera_height_mm: float,
    crop: str | None = None,
    tile_mm: float = DEFAULT_TILE_MM,
    overlap: float = 0.25,
    hfov_deg: float = HFOV_DEG,
    vfov_deg: float = VFOV_DEG,
) -> list[Tile]:
    """Recortes quadrados de ``tile_mm`` de planta, cobrindo o quadro.

    Args:
        frame_size: ``(largura, altura)`` do quadro em pixels.
        distance_mm: distância até a planta — a mediana da profundidade da OAK-D
            na região de interesse.
        camera_height_mm: altura da câmera em relação ao chão.
        crop: ``"banana"``, ``"citrus"`` ou ``"coffee"`` para marcar as zonas; ``None`` não marca.

    Raises:
        ValueError: se a geometria não fizer sentido ou o recorte ficar pequeno
            demais para o modelo (ver :data:`MIN_TILE_PX`).
    """
    width, height = frame_size
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap precisa ficar em [0, 1) (é {overlap}).")
    gsd_x = gsd_mm_per_px(distance_mm, hfov_deg, width)
    gsd_y = gsd_mm_per_px(distance_mm, vfov_deg, height)
    side = int(round(tile_mm / max(gsd_x, gsd_y)))
    if side < MIN_TILE_PX:
        raise ValueError(
            f"A {distance_mm:.0f} mm, {tile_mm:.0f} mm de planta cabem em {side} px — "
            f"menos que {MIN_TILE_PX}. Aproxime a câmera ou aumente tile_mm."
        )
    side = min(side, width, height)
    step = max(1, int(round(side * (1.0 - overlap))))

    tiles = []
    for y0 in _starts(height, side, step):
        center_y = y0 + side / 2.0
        # Pinhole: acima do centro óptico, mais alto que a câmera.
        center_height = camera_height_mm + (height / 2.0 - center_y) * gsd_y
        if center_height < 0:
            continue  # abaixo do chão: é solo, não planta
        zones = zones_at(crop, center_height) if crop else ()
        for x0 in _starts(width, side, step):
            tiles.append(Tile((x0, y0, x0 + side, y0 + side), round(center_height, 1), zones))
    return tiles


def zones_at(crop: str, height_mm: float) -> tuple[str, ...]:
    """As zonas da cultura que contêm uma altura.

    Raises:
        KeyError: se a cultura não tiver zonas definidas.
    """
    return tuple(name for name, low, high in ZONES[crop] if low <= height_mm <= high)


def vertical_coverage_mm(
    camera_height_mm: float, distance_mm: float, vfov_deg: float = VFOV_DEG
) -> tuple[float, float]:
    """Faixa de alturas que um andar enxerga, com o chão como limite inferior."""
    half = distance_mm * math.tan(math.radians(vfov_deg) / 2.0)
    return max(0.0, camera_height_mm - half), camera_height_mm + half


def _starts(total: int, side: int, step: int) -> list[int]:
    """Inícios dos recortes ao longo de um eixo, com o último encostado na borda."""
    if side >= total:
        return [0]
    starts = list(range(0, total - side + 1, step))
    if starts[-1] != total - side:
        starts.append(total - side)
    return starts
