"""Correção da distorção da lente: a foto vira a de uma câmera ideal (pinhole).

A OAK-D Pro W tem lente de ~127°. Nas bordas de uma lente assim a escala não é a do
centro — e a Edge AI corta o quadro em recortes de **150 mm** pela escala (mm por pixel).
Sem correção, o recorte da borda cobre mais (ou menos) planta do que diz, e a folha chega
ao modelo num tamanho que ele não viu no treino.

Cada OAK-D sai de fábrica calibrada: intrínsecos (``fx``, ``fy``, ``cx``, ``cy``) e
coeficientes de distorção gravados no próprio aparelho (``readCalibration``). Aqui se
monta uma vez o mapa "pixel da foto ideal → pixel da foto da lente" e cada quadro é
remapeado. A foto ideal mantém a distância focal do centro: a escala vira exata em
qualquer ponto (``mm/px = distância / fx``), e o campo de visão **efetivo**
(``2·atan(W / 2fx)``) vai junto com o quadro — é ele que a Edge AI e a nuvem usam para
cortar e posicionar os recortes, no lugar dos 127° nominais. As bordas que uma câmera
ideal não alcança saem da foto (pretas), em vez de esticadas.

Modelos: ``perspective`` (Brown-Conrady racional: k1..k6, p1, p2 — o padrão do
OpenCV/depthai) e ``fisheye`` (Kannala-Brandt: k1..k4). Usa o OpenCV quando instalado
(``cv2.remap``, rápido na Jetson); sem ele, a mesma conta em NumPy (testes e bancada).
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

MODELOS = ("perspective", "fisheye")


@dataclass(frozen=True)
class LensModel:
    """Intrínsecos e distorção de uma câmera, na resolução da foto."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    model: str = "perspective"
    coeffs: tuple[float, ...] = ()
    """``perspective``: k1, k2, p1, p2, k3, k4, k5, k6 (o resto é ignorado);
    ``fisheye``: k1, k2, k3, k4."""

    def __post_init__(self) -> None:
        if self.model not in MODELOS:
            raise ValueError(f"modelo de lente {self.model!r}: use {', '.join(MODELOS)}")
        if self.width <= 0 or self.height <= 0 or self.fx <= 0 or self.fy <= 0:
            raise ValueError("intrínsecos inválidos")

    @classmethod
    def from_depthai(cls, calib: Any, dai: Any, socket: Any, width: int,
                     height: int) -> LensModel:
        """A calibração de fábrica gravada na OAK-D, na resolução da foto parada."""
        k = calib.getCameraIntrinsics(socket, width, height)
        coeffs = tuple(float(c) for c in calib.getDistortionCoefficients(socket))
        modelo = "perspective"
        try:
            if "fisheye" in str(calib.getDistortionModel(socket)).lower():
                modelo = "fisheye"
        except AttributeError:  # depthai antigo: só perspectiva
            pass
        return cls(width, height, float(k[0][0]), float(k[1][1]), float(k[0][2]),
                   float(k[1][2]), modelo, coeffs)

    def distort(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Coordenadas normalizadas ideais → normalizadas como a lente as forma."""
        c = list(self.coeffs) + [0.0] * 8
        if self.model == "fisheye":
            k1, k2, k3, k4 = c[:4]
            r = np.sqrt(x * x + y * y)
            theta = np.arctan(r)
            t2 = theta * theta
            theta_d = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
            escala = np.where(r > 1e-12, theta_d / np.maximum(r, 1e-12), 1.0)
            return x * escala, y * escala
        k1, k2, p1, p2, k3, k4, k5, k6 = c[:8]
        r2 = x * x + y * y
        radial = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1 + r2 * (k4 + r2 * (k5 + r2 * k6)))
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return xd, yd


@dataclass
class Undistorter:
    """Remapeia quadros da lente para a câmera ideal com a mesma distância focal."""

    lens: LensModel
    _mapas: dict[tuple[int, int], Any] = field(default_factory=dict, repr=False)

    @property
    def hfov_deg(self) -> float:
        """Campo de visão horizontal da foto corrigida (o que a Edge AI deve usar)."""
        return math.degrees(2 * math.atan(self.lens.width / (2 * self.lens.fx)))

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2 * math.atan(self.lens.height / (2 * self.lens.fy)))

    def _mapa(self, largura: int, altura: int):
        chave = (largura, altura)
        if chave not in self._mapas:
            lente = self.lens
            sx, sy = largura / lente.width, altura / lente.height  # outra resolução
            fx, fy, cx, cy = lente.fx * sx, lente.fy * sy, lente.cx * sx, lente.cy * sy
            u, v = np.meshgrid(np.arange(largura, dtype=np.float64),
                               np.arange(altura, dtype=np.float64))
            xd, yd = lente.distort((u - cx) / fx, (v - cy) / fy)
            mx = (xd * fx + cx).astype(np.float32)
            my = (yd * fy + cy).astype(np.float32)
            try:
                import cv2

                self._mapas[chave] = ("cv2", *cv2.convertMaps(mx, my, cv2.CV_16SC2))
            except ImportError:
                self._mapas[chave] = ("numpy", mx, my)
        return self._mapas[chave]

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        """``(H, W, 3)`` uint8 da lente → ``(H, W, 3)`` uint8 da câmera ideal."""
        altura, largura = rgb.shape[:2]
        tipo, a, b = self._mapa(largura, altura)
        if tipo == "cv2":
            import cv2

            return cv2.remap(rgb, a, b, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return _remap_bilinear(rgb, a, b)

    def apply_jpeg(self, dados: bytes, quality: int = 92) -> tuple[bytes, int, int]:
        """O JPEG da câmera, corrigido e recodificado: ``(bytes, largura, altura)``."""
        from PIL import Image

        with Image.open(io.BytesIO(dados)) as im:
            rgb = np.asarray(im.convert("RGB"))
        corrigida = Image.fromarray(self.apply(rgb))
        saida = io.BytesIO()
        corrigida.save(saida, "JPEG", quality=quality)
        return saida.getvalue(), corrigida.width, corrigida.height


def _remap_bilinear(rgb: np.ndarray, mx: np.ndarray, my: np.ndarray) -> np.ndarray:
    """``cv2.remap`` linear em NumPy; fora da foto, preto."""
    altura, largura = rgb.shape[:2]
    # A última linha/coluna também é da foto: o par vizinho é recuado para dentro e o
    # peso passa todo para o pixel da borda (fx = 1), em vez de virar preto.
    dentro = (mx >= 0) & (my >= 0) & (mx <= largura - 1) & (my <= altura - 1)
    x0c = np.clip(np.floor(mx).astype(np.int64), 0, largura - 2)
    y0c = np.clip(np.floor(my).astype(np.int64), 0, altura - 2)
    fx, fy = np.clip(mx - x0c, 0, 1)[..., None], np.clip(my - y0c, 0, 1)[..., None]
    img = rgb.astype(np.float32)
    topo = img[y0c, x0c] * (1 - fx) + img[y0c, x0c + 1] * fx
    base = img[y0c + 1, x0c] * (1 - fx) + img[y0c + 1, x0c + 1] * fx
    saida = topo * (1 - fy) + base * fy
    saida[~dentro] = 0
    return np.clip(saida + 0.5, 0, 255).astype(np.uint8)
