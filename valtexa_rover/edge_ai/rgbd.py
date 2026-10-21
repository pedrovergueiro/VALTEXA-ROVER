"""Fusão RGB + profundidade: o que no quadro é a planta, e o que é fundo.

A OAK-D entrega, junto da foto, um mapa de profundidade alinhado a ela (``Frame.depth``).
Até aqui ele só dava uma mediana — a distância usada para cortar recortes de 150 mm. Mas
num cafeeiro, pelos vãos do dossel aparecem a rua, a planta da linha vizinha e o céu: a
mediana puxa para longe, e os recortes desses vãos vão ao modelo como se fossem folha. O
modelo, treinado com folha, lê qualquer coisa ali — inclusive doença.

Este módulo faz três coisas com o mapa:

1. **A distância da planta** (:func:`plant_plane`): o pico mais próximo do histograma da
   profundidade no centro do quadro — a superfície grande mais perto da câmera é a saia da
   planta; o fundo visto pelos vãos fica em outro pico, mais longe. Em volta do pico, a
   faixa da planta (``frente_mm`` à frente, ``tras_mm`` atrás).
2. **Recorte a recorte** (:func:`tile_depths`): quanto do recorte tem medida, quanto disso
   está na faixa da planta e quanto está atrás dela. Recorte medido e fora da faixa é
   fundo: com ``profundidade.modo = filtrar`` ele não vai ao modelo nem à evidência.
   Recorte **sem** medida suficiente fica — não se descarta o que não se mediu.
3. **Abertura do dossel** (:func:`summary`): a fração dos pixels medidos dos recortes que
   está atrás da planta, no quadro e por terço. Planta que perde folha (ferrugem, seca)
   abre o dossel; é um indicador **não calibrado** — vai à nuvem para ser comparado com a
   avaliação do agrônomo, não decide nada aqui.

Os padrões (faixa, frações mínimas) são da geometria do café — câmera a ~27 cm da saia — e
**não foram medidos em campo**. O primeiro teste é na bancada (``valtexa-rover ensaio
profundidade``) e depois numa linha de café com o agrônomo.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from valtexa_rover.config import DepthConfig

BIN_MM = 25.0
"""Largura da faixa do histograma de profundidade."""

MEDIDA_UTIL_MM = (80.0, 8000.0)
"""Abaixo de 8 cm a OAK-D não mede (é ruído ou a própria tampa); acima de 8 m, o céu."""

PICO_MINIMO = 0.15
"""O pico da planta precisa ter ao menos esta fração do maior pico: uma folha solta na frente
da lente (fração ínfima do quadro) não é a planta; a saia que divide o centro com um vão
grande ainda é."""


@dataclass(frozen=True, slots=True)
class PlantPlane:
    """Onde está a planta neste quadro, pela profundidade."""

    distance_mm: float | None
    near_mm: float | None
    far_mm: float | None
    coverage: float
    """Fração dos pixels do centro do quadro com medida."""
    sufficient: bool
    """Medida suficiente para separar planta de fundo."""


@dataclass(frozen=True, slots=True)
class TileDepth:
    """A profundidade dentro de um recorte."""

    depth_mm: float | None
    valid: float
    """Fração dos pixels do recorte com medida."""
    on_plant: float | None
    """Fração dos pixels medidos dentro da faixa da planta."""
    behind: float | None
    """Fração dos pixels medidos atrás da faixa (vão do dossel)."""
    is_plant: bool | None
    """``None``: sem medida suficiente para julgar."""


def _medidos(valores: np.ndarray) -> np.ndarray:
    return (valores > MEDIDA_UTIL_MM[0]) & (valores < MEDIDA_UTIL_MM[1])


def plant_plane(depth: np.ndarray | None, cfg: DepthConfig, roi: float = 0.6) -> PlantPlane:
    """A distância da planta pelo pico mais próximo do histograma, no centro do quadro."""
    if depth is None:
        return PlantPlane(None, None, None, 0.0, False)
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim != 2 or d.size == 0:
        return PlantPlane(None, None, None, 0.0, False)
    h, w = d.shape
    dy, dx = int(h * (1 - roi) / 2), int(w * (1 - roi) / 2)
    centro = d[dy:h - dy, dx:w - dx]
    validos = _medidos(centro)
    cobertura = float(validos.mean()) if centro.size else 0.0
    if cobertura < cfg.min_cobertura_quadro or int(validos.sum()) < 16:
        return PlantPlane(None, None, None, cobertura, False)
    valores = centro[validos]
    bordas = np.arange(MEDIDA_UTIL_MM[0], MEDIDA_UTIL_MM[1] + BIN_MM, BIN_MM)
    hist, _ = np.histogram(valores, bordas)
    suave = np.convolve(hist.astype(np.float64), np.ones(3) / 3.0, mode="same")
    i = int(np.flatnonzero(suave >= PICO_MINIMO * suave.max())[0])
    while i + 1 < len(suave) and suave[i + 1] >= suave[i]:  # sobe até o topo do pico
        i += 1
    pico = (bordas[i] + bordas[i + 1]) / 2.0
    perto = valores[np.abs(valores - pico) <= max(2 * BIN_MM, 0.15 * pico)]
    distancia = float(np.median(perto)) if perto.size else float(pico)
    return PlantPlane(distancia, max(MEDIDA_UTIL_MM[0], distancia - cfg.frente_mm),
                      distancia + cfg.tras_mm, cobertura, True)


def tile_depths(depth: np.ndarray, image_size: tuple[int, int],
                boxes: Sequence[Sequence[float]], plane: PlantPlane,
                cfg: DepthConfig) -> list[TileDepth]:
    """A profundidade de cada recorte (caixas em pixels da foto; o mapa pode ser menor)."""
    d = np.asarray(depth, dtype=np.float64)
    altura_d, largura_d = d.shape
    largura, altura = image_size
    sx, sy = largura_d / float(largura), altura_d / float(altura)
    saida = []
    for x0, y0, x1, y1 in boxes:
        xa = min(max(int(np.floor(x0 * sx)), 0), largura_d - 1)
        ya = min(max(int(np.floor(y0 * sy)), 0), altura_d - 1)
        xb = max(xa + 1, min(int(np.ceil(x1 * sx)), largura_d))
        yb = max(ya + 1, min(int(np.ceil(y1 * sy)), altura_d))
        regiao = d[ya:yb, xa:xb]
        medidos = _medidos(regiao)
        fracao = float(medidos.mean())
        if not medidos.any():
            saida.append(TileDepth(None, 0.0, None, None, None))
            continue
        valores = regiao[medidos]
        if plane.sufficient:
            assert plane.near_mm is not None and plane.far_mm is not None
            na_faixa = (valores >= plane.near_mm) & (valores <= plane.far_mm)
            na_planta = float(na_faixa.mean())
            atras = float((valores > plane.far_mm).mean())
            julga = fracao >= cfg.min_validos_recorte
            e_planta = bool(na_planta >= cfg.min_na_planta) if julga else None
            prof = float(np.median(valores[na_faixa])) if na_faixa.any() else float(
                np.median(valores))
            saida.append(TileDepth(round(prof, 1), round(fracao, 4), round(na_planta, 4),
                                   round(atras, 4), e_planta))
        else:
            saida.append(TileDepth(round(float(np.median(valores)), 1), round(fracao, 4),
                                   None, None, None))
    return saida


def summary(plane: PlantPlane, tiles: Sequence[TileDepth], zones: Sequence[Sequence[str]],
            mode: str, excluded: int) -> dict:
    """O que vai no resultado (``rgbd``): a planta, a cobertura e a abertura do dossel."""
    abertura, por_zona = None, {}
    pesos = [(t.behind, t.valid, z) for t, z in zip(tiles, zones, strict=True)
             if t.behind is not None and t.valid > 0]
    if pesos:
        total = sum(v for _, v, _ in pesos)
        abertura = round(sum(b * v for b, v, _ in pesos) / total, 4)
        acumulado: dict[str, list[float]] = {}
        for b, v, zs in pesos:
            for zona in zs:
                acumulado.setdefault(zona, [0.0, 0.0])
                acumulado[zona][0] += b * v
                acumulado[zona][1] += v
        por_zona = {z: round(s / p, 4) for z, (s, p) in sorted(acumulado.items()) if p > 0}
    return {
        "mode": mode,
        "sufficient": plane.sufficient,
        "coverage": round(plane.coverage, 4),
        "plant_distance_mm": None if plane.distance_mm is None else round(plane.distance_mm, 1),
        "plant_near_mm": None if plane.near_mm is None else round(plane.near_mm, 1),
        "plant_far_mm": None if plane.far_mm is None else round(plane.far_mm, 1),
        "excluded_tiles": int(excluded),
        "canopy_openness": abertura,
        "canopy_openness_by_zone": por_zona,
        "calibrated": False,
    }
