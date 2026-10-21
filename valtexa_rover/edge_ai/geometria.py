"""Geometria de cada recorte pela profundidade: onde a lesão está e como a folha está virada.

O recorte deixa de ser só um retângulo da foto. Com o mapa de profundidade alinhado (OAK-D),
cada recorte ganha:

* o **ponto 3D** do centro, no referencial da câmera (``x`` para a direita da foto, ``y``
  para baixo, ``z`` para dentro da planta), pela câmera de furo com o campo de visão
  efetivo do quadro (o da lente corrigida) — o que a nuvem precisa para pôr a lesão no
  referencial da **planta** (ao longo da rua, altura, profundidade no dossel) e reencontrá-la
  na passada seguinte (``ia/app/plant_fusion.py``);
* a **orientação da superfície**: um plano ajustado (mínimos quadrados) aos pixels medidos
  do recorte, com a normal, a **inclinação** em relação ao eixo da câmera e o resíduo — folha
  de frente (0°), de lado (perto de 90°), ou superfície que nem é plana (resíduo alto: várias
  folhas, borda, vão);
* a **face foliar** fica ``indeterminada``. A profundidade dá a orientação da superfície, não
  qual face está para a câmera: a folha de café tem o limbo ondulado, a câmera vê de lado, e a
  mesma normal serve às duas faces. Decidir adaxial/abaxial pede outra evidência (textura,
  cor, nervura) num modelo próprio — até lá, o campo existe e diz que não se sabe.

Nada aqui muda a decisão do quadro: é o que vai junto com o recorte (``tiles[].geometria``).
**Não testado em hardware**: a calibração da OAK-D (alinhamento RGB x profundidade, campo de
visão) entra pela lente corrigida, mas só a bancada dirá o erro em milímetros.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

MEDIDA_UTIL_MM = (80.0, 6000.0)
MIN_PONTOS = 12
RESIDUO_PLANO_MM = 12.0
"""Acima disto o recorte não é uma superfície (várias folhas, borda, vão): a orientação não
vale e o campo diz isso."""
FACE_INDETERMINADA = "indeterminada"


def geometria_do_recorte(depth: np.ndarray, image_size: tuple[int, int],
                         box: Sequence[float], hfov_deg: float,
                         faixa_mm: tuple[float, float] | None = None,
                         max_pontos: int = 2000) -> dict | None:
    """O ponto 3D e a orientação de um recorte; ``None`` sem medida bastante.

    ``faixa_mm``: a faixa de profundidade da planta (:func:`.rgbd.plant_plane`) — o que está
    fora dela (o vão, o fundo) não entra no plano.
    """
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim != 2 or d.size == 0 or not 0 < hfov_deg < 180:
        return None
    altura_d, largura_d = d.shape
    largura, altura = image_size
    sx, sy = largura_d / float(largura), altura_d / float(altura)
    x0, y0, x1, y1 = box
    xa = min(max(math.floor(x0 * sx), 0), largura_d - 1)
    ya = min(max(math.floor(y0 * sy), 0), altura_d - 1)
    xb = max(xa + 1, min(math.ceil(x1 * sx), largura_d))
    yb = max(ya + 1, min(math.ceil(y1 * sy), altura_d))
    regiao = d[ya:yb, xa:xb]
    vv, uu = np.mgrid[ya:yb, xa:xb]
    ok = (regiao > MEDIDA_UTIL_MM[0]) & (regiao < MEDIDA_UTIL_MM[1])
    if faixa_mm is not None:
        ok &= (regiao >= faixa_mm[0]) & (regiao <= faixa_mm[1])
    medidos = float(ok.mean()) if ok.size else 0.0
    if int(ok.sum()) < MIN_PONTOS:
        return None
    z = regiao[ok]
    u = uu[ok].astype(np.float64) + 0.5
    v = vv[ok].astype(np.float64) + 0.5
    if len(z) > max_pontos:  # amostra regular: o plano não precisa de todos os pixels
        passo = math.ceil(len(z) / max_pontos)
        z, u, v = z[::passo], u[::passo], v[::passo]
    # Câmera de furo no mapa de profundidade (pixel quadrado; centro óptico no meio).
    f = largura_d / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    cx, cy = largura_d / 2.0, altura_d / 2.0
    x = (u - cx) * z / f
    y = (v - cy) * z / f
    # Centro do recorte: o centro geométrico da caixa, na mediana da profundidade medida.
    zc = float(np.median(z))
    uc, vc = (x0 + x1) / 2.0 * sx, (y0 + y1) / 2.0 * sy
    centro = [(uc - cx) * zc / f, (vc - cy) * zc / f, zc]
    # Plano z = a x + b y + c pelos mínimos quadrados; normal (-a, -b, 1), virada à câmera.
    a_mat = np.column_stack([x, y, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(a_mat, z, rcond=None)
    residuo = float(np.sqrt(np.mean((a_mat @ coef - z) ** 2)))
    normal = np.array([-coef[0], -coef[1], 1.0])
    normal = normal / np.linalg.norm(normal)
    if normal[2] > 0:  # a câmera olha para +z: a normal aponta para ela (-z)
        normal = -normal
    inclinacao = math.degrees(math.acos(min(1.0, abs(float(normal[2])))))
    plana = residuo <= RESIDUO_PLANO_MM
    return {
        "centro_mm": [round(c, 1) for c in centro],
        "normal": [round(float(c), 4) for c in normal] if plana else None,
        "inclinacao_graus": round(inclinacao, 1) if plana else None,
        "residuo_plano_mm": round(residuo, 2),
        "superficie": "plana" if plana else "irregular",
        "pixels_medidos": round(medidos, 4),
        "pontos": len(z),
        "face": FACE_INDETERMINADA,
    }
