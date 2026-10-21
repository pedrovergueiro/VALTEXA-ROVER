"""Qualidade da captura: a foto está boa o bastante para ser diagnosticada?

A detecção de novidade diz que uma imagem é estranha, mas não diz **por quê** — e
uma foto borrada, escura ou estourada pede uma ação diferente de uma doença
fora do catálogo: pede outra foto, não um agrônomo. O rover pode refazer a
captura na hora; o agrônomo não diagnostica o que não enxerga.

## O que é "boa o bastante"

Não há limiar universal de nitidez. Uma folha sadia e lisa tem menos borda que
uma folha com ferrugem; um dataset a 128x128 tem outra escala de detalhe que um
a 3000x2000. Um número fixo, tirado de tutorial, marcaria metade de um dataset e
nada do outro.

Então a régua é o **próprio dataset de treino do modelo**: uma foto é "borrada"
quando é mais borrada que quase tudo que o modelo viu ao aprender. Fora dessa
faixa a resposta dele é extrapolação, seja qual for a confiança. É o mesmo
raciocínio da novidade, só que sobre a captura em vez do conteúdo — e por isso
cada modelo tem a sua referência, gravada por ``training.fit_quality``.

## Medido sobre o que o modelo vê

As métricas saem do **lote já pré-processado** — a imagem redimensionada para a
entrada do modelo. Desfoque que some ao reduzir uma foto de 12 MP para 224x224
não importa para o modelo, e não é marcado; desfoque que sobrevive é o que ele
de fato recebe. De quebra, a medida não custa uma segunda decodificação.

Opcional como a novidade: sem a referência, a API sobe do mesmo jeito e
``quality`` vem nulo.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

DEFAULT_PERCENTILE = 0.5
"""Cauda do treino que vira limiar, em cada ponta.

0,5 significa: uma foto só é marcada como borrada se for mais borrada que 99,5%
das fotos de treino. Com quatro verificações, o alarme falso no próprio treino
fica perto de 2% — medido, não presumido, pelo ``fit_quality``.
"""

HIGHLIGHT_LEVEL = 250.0
"""Luminância a partir da qual um pixel conta como estourado, em ``[0, 255]``."""

LUMA = np.array([0.299, 0.587, 0.114])
"""Pesos da luminância (ITU-R BT.601), os mesmos do ``convert("L")`` do Pillow."""

BLURRY = "blurry"
DARK = "dark"
OVEREXPOSED = "overexposed"
LOW_CONTRAST = "low_contrast"

ISSUE_ACTIONS = {
    BLURRY: "borrada — segure a câmera ou reduza a velocidade do rover",
    DARK: "escura — mais luz ou exposição maior",
    OVEREXPOSED: "estourada — evite sol direto na lente ou reduza a exposição",
    LOW_CONTRAST: "sem contraste — lente suja, neblina ou o quadro não tem folha",
}
"""O que cada problema pede de quem está capturando."""


@dataclass(frozen=True)
class CaptureMetrics:
    """As medidas de uma imagem, todas na escala ``[0, 255]`` de luminância."""

    sharpness: float
    """Variância do laplaciano: quanta borda fina sobrou. Baixo = borrado."""

    brightness: float
    """Luminância média."""

    contrast: float
    """Desvio-padrão da luminância. Perto de zero = cor chapada."""

    highlights: float
    """Fração dos pixels estourados, acima de :data:`HIGHLIGHT_LEVEL`."""


def measure(batch: np.ndarray) -> list[CaptureMetrics]:
    """Mede um lote pré-processado ``(N, H, W, 3)`` na escala ``[-1, 1]``.

    Recebe o lote que vai para o modelo, não a imagem original — ver o topo do
    módulo.
    """
    array = np.asarray(batch, dtype=np.float64)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"Esperado lote (N, H, W, 3); recebido {array.shape}.")
    if array.shape[1] < 3 or array.shape[2] < 3:
        raise ValueError(f"Imagem pequena demais para medir nitidez: {array.shape[1:3]}.")

    # Desfaz o ``scale_to_unit_range`` para voltar à escala de pixel, em que os
    # números são os mesmos de qualquer ferramenta de fotografia.
    luminance = np.clip((array + 1.0) * 127.5, 0.0, 255.0) @ LUMA
    laplacian = (
        4.0 * luminance[:, 1:-1, 1:-1]
        - luminance[:, :-2, 1:-1]
        - luminance[:, 2:, 1:-1]
        - luminance[:, 1:-1, :-2]
        - luminance[:, 1:-1, 2:]
    )

    sharpness = laplacian.var(axis=(1, 2))
    brightness = luminance.mean(axis=(1, 2))
    contrast = luminance.std(axis=(1, 2))
    highlights = (luminance >= HIGHLIGHT_LEVEL).mean(axis=(1, 2))

    return [
        CaptureMetrics(float(s), float(b), float(c), float(h))
        for s, b, c, h in zip(sharpness, brightness, contrast, highlights, strict=True)
    ]


@dataclass(frozen=True)
class QualityReference:
    """Os limites da captura que um modelo conhece, tirados do treino dele."""

    img_size: int
    """Resolução em que as métricas foram medidas. Tem de bater com a do modelo:
    nitidez medida a 224 não se compara com nitidez medida a 384."""

    min_sharpness: float
    min_brightness: float
    max_brightness: float
    min_contrast: float
    max_highlights: float
    percentile: float
    fitted_on: int
    """Quantas imagens formaram a referência."""

    @classmethod
    def fit(
        cls,
        metrics: list[CaptureMetrics],
        img_size: int,
        percentile: float = DEFAULT_PERCENTILE,
    ) -> QualityReference:
        """Calibra os limites a partir das métricas do dataset de treino.

        Raises:
            ValueError: se não houver imagens bastantes para o percentil pedido.
        """
        if not 0.0 < percentile < 50.0:
            raise ValueError(f"percentile precisa ficar entre 0 e 50 (é {percentile}).")
        minimum = int(np.ceil(100.0 / percentile))
        if len(metrics) < minimum:
            raise ValueError(
                f"{len(metrics)} imagens não sustentam o percentil {percentile:g}: "
                f"a cauda teria menos de uma imagem. Use ao menos {minimum}."
            )

        def column(name: str) -> np.ndarray:
            return np.array([getattr(m, name) for m in metrics])

        low, high = percentile, 100.0 - percentile
        return cls(
            img_size=img_size,
            min_sharpness=float(np.percentile(column("sharpness"), low)),
            min_brightness=float(np.percentile(column("brightness"), low)),
            max_brightness=float(np.percentile(column("brightness"), high)),
            min_contrast=float(np.percentile(column("contrast"), low)),
            max_highlights=float(np.percentile(column("highlights"), high)),
            percentile=percentile,
            fitted_on=len(metrics),
        )

    def assess(self, metrics: CaptureMetrics) -> list[str]:
        """Os problemas de uma imagem, na ordem em que pedem ação.

        Escura e estourada são decididas pela luminância média *e* pelos pixels
        estourados: uma foto com o céu queimado num canto e a folha bem exposta
        tem média normal, mas perdeu parte do quadro.
        """
        issues = []
        if metrics.sharpness < self.min_sharpness:
            issues.append(BLURRY)
        if metrics.brightness < self.min_brightness:
            issues.append(DARK)
        if metrics.brightness > self.max_brightness or metrics.highlights > self.max_highlights:
            issues.append(OVEREXPOSED)
        if metrics.contrast < self.min_contrast:
            issues.append(LOW_CONTRAST)
        return issues

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> QualityReference:
        """Lê a referência gravada pelo ``fit_quality``.

        Raises:
            ValueError: se o arquivo estiver ilegível ou incompleto.
        """
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return cls(**raw)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"Referência de qualidade ilegível em {path}: {exc}") from exc
