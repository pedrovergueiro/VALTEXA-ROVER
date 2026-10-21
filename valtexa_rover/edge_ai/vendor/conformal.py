"""Predição conformal: um conjunto de classes com garantia de cobertura.

Um softmax devolve **uma** classe, e a confiança que vem junto não é uma
probabilidade calibrada — o modelo diz 0,93 e erra. Quando `Blight` e
`Gray_Leaf_Spot` se confundem (e se confundem: são o par dos 88 grupos de rótulo
contraditório), forçar uma resposta só produz um erro confiante.

A predição conformal troca a pergunta. Em vez de "qual é a doença?", responde
"quais doenças **não dá para descartar**?", com uma garantia que não depende de
o modelo estar bem calibrado:

    P(classe verdadeira ∈ conjunto) >= 1 - alpha

Com ``alpha = 0,10`` (o padrão), o conjunto contém a doença certa em pelo menos
90% das imagens — e isso vale **para cada classe separadamente**, porque os limiares são
calibrados por classe (conformal de Mondrian). Sem essa separação, a garantia
valeria na média, e a classe difícil (`Gray_Leaf_Spot`, recall 0,85) pagaria
pela fácil.

## Como ler o conjunto

- **uma classe**: o modelo descartou todas as outras com a garantia pedida;
- **duas ou mais**: é a dúvida real do modelo, dita com precisão — a folha é
  `Blight` ou `Gray_Leaf_Spot`, e o agrônomo sabe entre o quê decidir;
- **vazio**: nenhuma classe alcança a confiança que 95% das imagens verdadeiras
  dela alcançam. Acontece quando a imagem não se parece com nenhuma doença
  conhecida — o território da novidade, por outro caminho —, mas também quando a
  probabilidade se divide entre classes que o modelo costuma responder com
  certeza. Nos dois casos a leitura é a mesma: inconclusivo.

## A garantia, e o que ela exige

Vale sob **permutabilidade**: as imagens de calibração e as de produção vêm da
mesma distribuição. Calibrado em dataset público, garante cobertura em dataset
público. Em campo, a cobertura precisa ser remedida com imagens da frota — é a
mesma ressalva de todas as métricas do projeto, com a vantagem de que aqui ela
tem nome e fórmula.

Referência: Sadinle, Lei & Wasserman (2019), *Least Ambiguous Set-Valued
Classifiers with Bounded Error Levels*; Vovk et al. (2005) para o caso Mondrian.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_ALPHA = 0.10
"""Erro tolerado: o conjunto deixa a classe certa de fora em no máximo 10%.

Escolhido por medição, em 26/09/2026. Num modelo que já acerta 95%, ``0,05`` quase
só devolve resposta única: sinalizava 37% dos erros e 5% de uma doença nunca vista.
Com ``0,10``, a resposta única acerta 98,5% no ``leaf-foliar``, 67% dos erros viram
conjunto ambíguo ou vazio, e um terço da doença nunca vista é interceptado — ao
custo de ~10% das imagens irem para o agrônomo como inconclusivas.
"""


def nonconformity(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Quão estranha é a classe verdadeira para o modelo: ``1 - p(verdadeira)``."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    return 1.0 - probabilities[np.arange(len(labels)), labels]


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """O quantil com a correção de amostra finita que dá a garantia.

    Com ``n`` escores, usa o de posição ``ceil((n + 1)(1 - alpha))``. Quando essa
    posição passa de ``n`` — poucas imagens para o alpha pedido —, não há
    limiar que garanta a cobertura, e a classe entra sempre (limiar 1,0). É o
    comportamento honesto: sem amostra, nada é descartado.
    """
    scores = np.sort(np.asarray(scores, dtype=np.float64))
    n = len(scores)
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if n == 0 or rank > n:
        return 1.0
    return float(scores[rank - 1])


@dataclass(frozen=True)
class ConformalReference:
    """Os limiares por classe, calibrados em imagens que o treino não viu."""

    class_names: tuple[str, ...]
    alpha: float
    thresholds: dict[str, list[float]]
    """Um limiar por classe para cada modo de inferência: ``"plain"`` e ``"tta"``.

    Os escores com TTA são a média de quatro orientações e têm outra
    distribuição; um limiar calibrado num modo não vale no outro.
    """

    calibrated_on: int
    class_counts: list[int]
    """Imagens de calibração por classe — abaixo de ``1/alpha`` a classe nunca é
    descartada (ver :func:`conformal_quantile`)."""

    field_calibrated_on: int = 0
    """Recortes de campo que alargaram os limiares (ver :meth:`widened`); 0 sem campo."""

    base_thresholds: dict[str, list[float]] | None = None
    """Os limiares da calibração original, antes de :meth:`widened`; ``None`` se nunca alargou.

    Existem porque o conjunto honesto (alargado) e o conjunto que a fusão por planta
    espera são coisas diferentes: a fusão divide a evidência de cada recorte pelo
    tamanho do conjunto, e essa divisão foi ajustada com os conjuntos da calibração
    original. Só alargar derrubou a detecção da banana de 9 para 7 de 18 plantas
    doentes (01/10/2026). A resposta da API traz o conjunto honesto; a fusão segue
    recebendo o original (``prediction_sets(..., base=True)``)."""

    @classmethod
    def fit(
        cls,
        probabilities: dict[str, np.ndarray],
        labels: np.ndarray,
        class_names: tuple[str, ...],
        alpha: float = DEFAULT_ALPHA,
    ) -> ConformalReference:
        """Calibra a partir das probabilidades de um conjunto rotulado.

        Args:
            probabilities: ``{"plain": (N, C), "tta": (N, C)}`` — as saídas do
                modelo nos dois modos, para as mesmas imagens.

        Raises:
            ValueError: se as formas não baterem ou o alpha não fizer sentido.
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha precisa ficar entre 0 e 1 (é {alpha}).")
        labels = np.asarray(labels, dtype=int)
        thresholds = {}
        for mode, probs in probabilities.items():
            probs = np.asarray(probs, dtype=np.float64)
            if probs.shape != (len(labels), len(class_names)):
                raise ValueError(
                    f"Probabilidades {mode!r} com forma {probs.shape}; esperado "
                    f"({len(labels)}, {len(class_names)})."
                )
            scores = nonconformity(probs, labels)
            thresholds[mode] = [
                conformal_quantile(scores[labels == k], alpha) for k in range(len(class_names))
            ]
        return cls(
            class_names=tuple(class_names),
            alpha=alpha,
            thresholds=thresholds,
            calibrated_on=len(labels),
            class_counts=[int((labels == k).sum()) for k in range(len(class_names))],
        )

    def widened(
        self, probabilities: dict[str, np.ndarray], labels: np.ndarray
    ) -> ConformalReference:
        """Une esta calibração a uma de campo, classe a classe, pelo maior limiar.

        A garantia vale enquanto as fotos vierem da distribuição da calibração. A
        validação é quase toda de dataset público; no campo, a banana cobria 57% em
        vez de 90% (medido em 30/09/2026). Com o maior dos dois limiares, cada classe
        presente no campo recebe a cobertura pedida nos dois domínios, cada um medido
        no seu — em esperança: com poucos recortes ela oscila (154 recortes deram 87%
        no campo, para 90% pedidos). O preço é conjunto maior onde o modelo duvida,
        que é a resposta certa. Classe sem exemplo no campo fica com o limiar de antes.

        Args:
            probabilities: ``{"plain": (N, C), "tta": (N, C)}`` dos recortes de campo.
        """
        labels = np.asarray(labels, dtype=int)
        thresholds = {}
        for mode, limits in self.thresholds.items():
            limits = list(limits)
            if mode in probabilities:
                scores = nonconformity(probabilities[mode], labels)
                for k in sorted(set(labels.tolist())):
                    field = conformal_quantile(scores[labels == k], self.alpha)
                    limits[k] = max(limits[k], field)
            thresholds[mode] = limits
        return dataclasses.replace(
            self,
            thresholds=thresholds,
            field_calibrated_on=len(labels),
            base_thresholds=self.base_thresholds or self.thresholds,
        )

    @property
    def coverage(self) -> float:
        return 1.0 - self.alpha

    def prediction_sets(
        self, probabilities: np.ndarray, tta: bool, *, base: bool = False
    ) -> list[list[str]]:
        """O conjunto de cada imagem, da classe mais provável para a menos.

        Args:
            base: usa os limiares da calibração original (antes do alargamento com o
                campo), que é o que a fusão por planta espera. Sem alargamento, é igual.

        Raises:
            ValueError: se o modo pedido não foi calibrado.
        """
        mode = "tta" if tta else "plain"
        source = self.base_thresholds if base and self.base_thresholds else self.thresholds
        if mode not in source:
            raise ValueError(f"Referência conformal sem calibração para o modo {mode!r}.")
        limits = np.asarray(source[mode])
        probabilities = np.asarray(probabilities, dtype=np.float64)
        included = (1.0 - probabilities) <= limits
        sets = []
        for row, keep in zip(probabilities, included, strict=True):
            order = [k for k in np.argsort(-row) if keep[k]]
            sets.append([self.class_names[k] for k in order])
        return sets

    def save(self, path: Path) -> None:
        payload = {
            "class_names": list(self.class_names),
            "alpha": self.alpha,
            "thresholds": self.thresholds,
            "calibrated_on": self.calibrated_on,
            "class_counts": self.class_counts,
            "field_calibrated_on": self.field_calibrated_on,
        }
        if self.base_thresholds is not None:
            payload["base_thresholds"] = self.base_thresholds
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ConformalReference:
        """Lê a referência gravada pelo ``fit_conformal``.

        Raises:
            ValueError: se o arquivo estiver ilegível ou incompleto.
        """
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            reference = cls(
                class_names=tuple(raw["class_names"]),
                alpha=float(raw["alpha"]),
                thresholds={mode: list(map(float, v)) for mode, v in raw["thresholds"].items()},
                calibrated_on=int(raw["calibrated_on"]),
                class_counts=list(map(int, raw["class_counts"])),
                field_calibrated_on=int(raw.get("field_calibrated_on", 0)),
                base_thresholds=(
                    {mode: list(map(float, v)) for mode, v in raw["base_thresholds"].items()}
                    if raw.get("base_thresholds")
                    else None
                ),
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Referência conformal ilegível em {path}: {exc}") from exc
        for family in (reference.thresholds, reference.base_thresholds or {}):
            for mode, limits in family.items():
                if len(limits) != len(reference.class_names):
                    raise ValueError(f"Referência conformal em {path}: modo {mode!r} incompleto.")
        return reference


def coverage_report(
    reference: ConformalReference, probabilities: np.ndarray, labels: np.ndarray, tta: bool
) -> dict:
    """O que a garantia virou num conjunto rotulado que a calibração não viu."""
    labels = np.asarray(labels, dtype=int)
    sets = reference.prediction_sets(probabilities, tta)
    truth = [reference.class_names[k] for k in labels]
    covered = np.array([t in s for t, s in zip(truth, sets, strict=True)])
    sizes = np.array([len(s) for s in sets])
    top1 = np.asarray(probabilities).argmax(axis=1)
    wrong = top1 != labels
    singleton = sizes == 1
    return {
        "coverage": float(covered.mean()),
        "coverage_per_class": {
            name: float(covered[labels == k].mean()) if (labels == k).any() else None
            for k, name in enumerate(reference.class_names)
        },
        "mean_set_size": float(sizes.mean()),
        "singleton_rate": float(singleton.mean()),
        "singleton_accuracy": float((~wrong)[singleton].mean()) if singleton.any() else None,
        "empty_rate": float((sizes == 0).mean()),
        "errors": int(wrong.sum()),
        "errors_flagged": float((~singleton)[wrong].mean()) if wrong.any() else None,
        "errors_with_truth_in_set": float(covered[wrong].mean()) if wrong.any() else None,
    }
