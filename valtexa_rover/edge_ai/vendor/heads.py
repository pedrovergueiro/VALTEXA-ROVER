"""Cabeças sobre o embedding do portão: uma segunda leitura da foto, quase de graça.

O portão de enquadramento (:mod:`app.framing`) já passa **toda** foto e todo recorte
do rover pela torre de visão do SigLIP 2, um modelo treinado em bilhões de pares
imagem-texto. O embedding que sai dela sabe muito mais sobre folhas do que os
datasets públicos ensinaram às nossas CNNs. Uma rede pequena sobre ele custa uma
multiplicação de matrizes — a torre já rodou.

Duas cabeças, ambas opcionais e por modelo, como a calibração conformal:

- ``<pesos>.head.npz`` — **diagnóstico**: as mesmas classes do modelo. A resposta
  passa a ser a mistura da CNN com ela (:meth:`DiagnosisHead.combine`).
- ``<pesos>.species.npz`` — **espécie**: a folha é da cultura do modelo? Abaixo do
  limiar, o portão recusa a foto (:meth:`SpeciesCheck.matches`).

## De onde vêm

O ``training.fit_heads`` calcula os embeddings do dataset do modelo (mesmo corte do
treino) e entrega ao **dNATY** (``dnaty.compress``), que busca por evolução a
arquitetura do MLP — largura e profundidade das camadas, ativação, saltos —,
escolhe pela validação e faz o ajuste fino. O resultado é exportado em matrizes:
a API não importa torch nem dNATY para usá-lo.

## A conta

A rede do dNATY é uma sequência de ``Linear -> BatchNorm -> ativação``, com saltos
opcionais entre camadas, e uma ``Linear`` final. Em inferência o BatchNorm é afim,
então vem dobrado na ``Linear`` anterior. A entrada é o embedding normalizado (L2)
e padronizado pela média e desvio do treino — os dois vão no arquivo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _erf(x: np.ndarray) -> np.ndarray:
    """``erf`` de Abramowitz-Stegun (7.1.26, erro < 1,5e-7): a GELU exata sem scipy."""
    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = (
        (((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592
    ) * t
    return sign * (1.0 - poly * np.exp(-x * x))


ACTIVATIONS = {
    "relu": lambda x: np.maximum(x, 0.0),
    "tanh": np.tanh,
    "gelu": lambda x: 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0))),
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
}
"""As ativações que o dNATY sabe sortear (``dnaty.core.arch.ACTIVATIONS``)."""


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class EmbeddingMLP:
    """O MLP do dNATY em matrizes, com o BatchNorm já dobrado."""

    mean: np.ndarray
    std: np.ndarray
    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]
    activations: tuple[str, ...]
    out_weight: np.ndarray
    out_bias: np.ndarray
    skips: tuple[tuple[int, int, int], ...] = ()
    """``(origem, destino, projeção)``: a saída da camada ``origem`` soma na
    ``destino`` depois da ativação; projeção ``-1`` é a identidade."""

    skip_weights: tuple[np.ndarray, ...] = ()
    skip_biases: tuple[np.ndarray, ...] = ()

    @property
    def input_size(self) -> int:
        return len(self.mean)

    @property
    def layer_sizes(self) -> list[int]:
        return [self.input_size] + [len(b) for b in self.biases] + [len(self.out_bias)]

    def logits(self, embeddings: np.ndarray) -> np.ndarray:
        """``(N, D)`` embeddings crus da torre de visão -> ``(N, C)`` logits."""
        x = np.asarray(embeddings, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.input_size:
            raise ValueError(
                f"Embedding de forma {x.shape}; a cabeça espera (N, {self.input_size})."
            )
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        outputs = [(x - self.mean) / self.std]
        for i, (weight, bias, name) in enumerate(
            zip(self.weights, self.biases, self.activations, strict=True)
        ):
            hidden = ACTIVATIONS[name](outputs[-1] @ weight.T + bias)
            for source, target, projection in self.skips:
                if target != i + 1 or source >= len(outputs):
                    continue
                shortcut = outputs[source]
                if projection >= 0:
                    shortcut = (
                        shortcut @ self.skip_weights[projection].T + self.skip_biases[projection]
                    )
                if shortcut.shape == hidden.shape:
                    hidden = hidden + shortcut
            outputs.append(hidden)
        return outputs[-1] @ self.out_weight.T + self.out_bias

    def probabilities(self, embeddings: np.ndarray) -> np.ndarray:
        return softmax(self.logits(embeddings))

    def to_arrays(self) -> dict[str, np.ndarray]:
        arrays = {
            "mean": self.mean,
            "std": self.std,
            "activations": np.array(self.activations),
            "out_weight": self.out_weight,
            "out_bias": self.out_bias,
            "skips": np.array(self.skips, dtype=np.int64).reshape(-1, 3),
        }
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases, strict=True)):
            arrays[f"weight_{i}"], arrays[f"bias_{i}"] = weight, bias
        for i, (weight, bias) in enumerate(zip(self.skip_weights, self.skip_biases, strict=True)):
            arrays[f"skip_weight_{i}"], arrays[f"skip_bias_{i}"] = weight, bias
        return arrays

    @classmethod
    def from_arrays(cls, raw) -> EmbeddingMLP:
        activations = tuple(str(a) for a in raw["activations"])
        unknown = sorted(set(activations) - set(ACTIVATIONS))
        if unknown:
            raise ValueError(f"ativação desconhecida: {', '.join(unknown)}")
        skips = tuple(tuple(int(v) for v in row) for row in raw["skips"])
        projections = 1 + max((s[2] for s in skips), default=-1)
        return cls(
            mean=raw["mean"].astype(np.float64),
            std=raw["std"].astype(np.float64),
            weights=tuple(raw[f"weight_{i}"].astype(np.float64) for i in range(len(activations))),
            biases=tuple(raw[f"bias_{i}"].astype(np.float64) for i in range(len(activations))),
            activations=activations,
            out_weight=raw["out_weight"].astype(np.float64),
            out_bias=raw["out_bias"].astype(np.float64),
            skips=skips,
            skip_weights=tuple(
                raw[f"skip_weight_{i}"].astype(np.float64) for i in range(projections)
            ),
            skip_biases=tuple(raw[f"skip_bias_{i}"].astype(np.float64) for i in range(projections)),
        )

    @classmethod
    def from_dnaty(cls, model, mean: np.ndarray, std: np.ndarray) -> EmbeddingMLP:
        """Converte a ``DynamicMLP`` que o ``dnaty.compress`` devolve (só no treino)."""
        import torch

        def array(tensor) -> np.ndarray:
            return tensor.detach().cpu().double().numpy()

        net = model.net
        hidden = len(model.layer_sizes) - 1
        weights, biases, activations = [], [], []
        for i in range(hidden):
            linear, norm = net[3 * i], net[3 * i + 1]
            scale = array(norm.weight / torch.sqrt(norm.running_var + norm.eps))
            weights.append(array(linear.weight) * scale[:, None])
            biases.append(
                (array(linear.bias) - array(norm.running_mean)) * scale + array(norm.bias)
            )
            activations.append(model.activations[i] if i < len(model.activations) else "relu")
        skips = tuple(
            (int(source), int(target), -1 if projection is None else int(projection))
            for source, target, projection in model.skip_connections
        )
        return cls(
            mean=np.asarray(mean, dtype=np.float64),
            std=np.asarray(std, dtype=np.float64),
            weights=tuple(weights),
            biases=tuple(biases),
            activations=tuple(activations),
            out_weight=array(net[3 * hidden].weight),
            out_bias=array(net[3 * hidden].bias),
            skips=skips,
            skip_weights=tuple(array(p.weight) for p in model.skip_projs),
            skip_biases=tuple(array(p.bias) for p in model.skip_projs),
        )


def _normalized(embeddings: np.ndarray) -> np.ndarray:
    x = np.asarray(embeddings, dtype=np.float64)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


@dataclass(frozen=True)
class DiagnosisHead:
    """``<pesos>.head.npz``: as classes do modelo, lidas do embedding do portão."""

    class_names: tuple[str, ...]
    mlp: EmbeddingMLP
    encoder: str
    """``model_id`` da torre de visão. Embedding de outra torre é outro espaço."""

    mix: dict[str, float]
    """Peso de cada leitura na média geométrica: ``cnn``, ``head`` e, se houver
    frases, ``zero_shot``. Peso zero tira a leitura da conta."""

    text_embeddings: np.ndarray | None = None
    """``(C, D)`` normalizados: uma frase (média de frases) por classe, na ordem de
    ``class_names`` — a leitura zero-shot, que não aprendeu nada dos nossos dados."""

    logit_scale: float = 0.0
    trained_on: int = 0
    license: str = ""

    def readings(self, embeddings: np.ndarray) -> dict[str, np.ndarray]:
        """As leituras que saem do embedding: a cabeça treinada e, se houver, o zero-shot."""
        out = {"head": self.mlp.probabilities(embeddings)}
        if self.text_embeddings is not None and self.mix.get("zero_shot", 0.0) > 0:
            out["zero_shot"] = softmax(
                self.logit_scale * _normalized(embeddings) @ self.text_embeddings.T
            )
        return out

    def combine(self, cnn_probabilities: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """Média geométrica ponderada das leituras, renormalizada.

        Geométrica e não aritmética: uma leitura que dá quase zero a uma classe
        derruba essa classe, mesmo que outra leitura a ame — é o que impede a CNN,
        confiante no estilo de foto que decorou, de passar por cima das outras duas.
        """
        readings = self.readings(embeddings)
        readings["cnn"] = np.asarray(cnn_probabilities, dtype=np.float64)
        used = {
            name: weight for name, weight in self.mix.items() if weight > 0 and name in readings
        }
        total = sum(used.values())
        log_mix = sum(
            weight * np.log(np.clip(readings[name], 1e-12, 1.0)) for name, weight in used.items()
        )
        return softmax(log_mix / total)

    def save(self, path: Path) -> None:
        arrays = self.mlp.to_arrays()
        if self.text_embeddings is not None:
            arrays["text_embeddings"] = self.text_embeddings
        np.savez(
            path,
            **arrays,
            class_names=np.array(self.class_names),
            encoder=np.array(self.encoder),
            mix_names=np.array(sorted(self.mix)),
            mix_weights=np.array([self.mix[name] for name in sorted(self.mix)]),
            logit_scale=np.array(self.logit_scale),
            trained_on=np.array(self.trained_on),
            license=np.array(self.license),
        )

    @classmethod
    def load(cls, path: Path) -> DiagnosisHead:
        """Raises: ValueError se o arquivo estiver ilegível ou incoerente."""
        try:
            with np.load(path, allow_pickle=False) as raw:
                mlp = EmbeddingMLP.from_arrays(raw)
                head = cls(
                    class_names=tuple(str(c) for c in raw["class_names"]),
                    mlp=mlp,
                    encoder=str(raw["encoder"]),
                    mix={
                        str(n): float(w)
                        for n, w in zip(raw["mix_names"], raw["mix_weights"], strict=True)
                    },
                    text_embeddings=(
                        raw["text_embeddings"].astype(np.float64)
                        if "text_embeddings" in raw
                        else None
                    ),
                    logit_scale=float(raw["logit_scale"]),
                    trained_on=int(raw["trained_on"]),
                    license=str(raw["license"]),
                )
        except (OSError, KeyError, ValueError) as exc:
            raise ValueError(f"Cabeça de diagnóstico ilegível em {path}: {exc}") from exc
        if len(head.mlp.out_bias) != len(head.class_names):
            raise ValueError(
                f"Cabeça em {path}: {len(head.mlp.out_bias)} saídas para "
                f"{len(head.class_names)} classes."
            )
        if sum(w for w in head.mix.values() if w > 0) <= 0:
            raise ValueError(f"Cabeça em {path}: mistura sem nenhum peso positivo.")
        return head


@dataclass(frozen=True)
class SpeciesCheck:
    """``<pesos>.species.npz``: a folha é da cultura do modelo, ou de outra planta?"""

    crop: str
    mlp: EmbeddingMLP
    """Duas saídas: ``[outra planta, a cultura]``."""

    threshold: float
    """Probabilidade mínima de ser a cultura para a foto passar."""

    encoder: str
    trained_on: int = 0
    license: str = ""

    def scores(self, embeddings: np.ndarray) -> np.ndarray:
        return self.mlp.probabilities(embeddings)[:, 1]

    def matches(self, embeddings: np.ndarray) -> np.ndarray:
        return self.scores(embeddings) >= self.threshold

    def save(self, path: Path) -> None:
        np.savez(
            path,
            **self.mlp.to_arrays(),
            crop=np.array(self.crop),
            threshold=np.array(self.threshold),
            encoder=np.array(self.encoder),
            trained_on=np.array(self.trained_on),
            license=np.array(self.license),
        )

    @classmethod
    def load(cls, path: Path) -> SpeciesCheck:
        """Raises: ValueError se o arquivo estiver ilegível ou incoerente."""
        try:
            with np.load(path, allow_pickle=False) as raw:
                check = cls(
                    crop=str(raw["crop"]),
                    mlp=EmbeddingMLP.from_arrays(raw),
                    threshold=float(raw["threshold"]),
                    encoder=str(raw["encoder"]),
                    trained_on=int(raw["trained_on"]),
                    license=str(raw["license"]),
                )
        except (OSError, KeyError, ValueError) as exc:
            raise ValueError(f"Verificação de espécie ilegível em {path}: {exc}") from exc
        if len(check.mlp.out_bias) != 2:
            raise ValueError(f"Verificação de espécie em {path}: precisa de 2 saídas.")
        return check
