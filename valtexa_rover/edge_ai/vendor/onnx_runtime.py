"""Inferência pelo ONNX Runtime: os mesmos pesos, sem torch nem Keras.

O ``.keras`` precisa do Keras e de um backend inteiro (o torch, com CUDA, passa
de 2 GB instalado) só para multiplicar matrizes na hora de servir. O
``training.export_onnx`` grava o mesmo modelo em ONNX — padrão aberto de grafo —
e o ONNX Runtime o executa com uma dependência de dezenas de MB, em CPU, GPU ou
num computador de bordo na ground station.

O arquivo exportado tem **duas saídas**, probabilidades e embedding, exatamente
como o modelo de duas saídas que :mod:`app.model` monta: a novidade continua
saindo da mesma passada, sem custo extra.

Liga com ``INFERENCE_RUNTIME=onnx``. Cada modelo cai para o Keras sozinho se o
``.onnx`` não existir ao lado dos pesos — nenhum modelo fica sem servir porque
outro foi exportado.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class OnnxModel:
    """Um grafo ONNX com a superfície que :class:`app.model.CornDiseaseModel` consome.

    Expõe ``input_shape``/``output_shape`` e é chamável como um modelo Keras —
    por isso passa pelo mesmo ``attach`` e pelas mesmas checagens de forma e de
    rótulos que os pesos ``.keras``.
    """

    def __init__(self, path: Path, providers: list[str] | None = None) -> None:
        import onnxruntime as ort  # import tardio: só quem escolhe ONNX precisa dele

        available = ort.get_available_providers()
        wanted = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.path = path
        self._session = ort.InferenceSession(
            str(path), providers=[p for p in wanted if p in available] or available
        )
        self._input = self._session.get_inputs()[0].name
        outputs = self._session.get_outputs()
        self.input_shape = _shape(self._session.get_inputs()[0].shape)
        self.output_shape = _shape(outputs[0].shape)
        self.has_embedding = len(outputs) > 1

    @property
    def providers(self) -> list[str]:
        return self._session.get_providers()

    def run(self, batch: np.ndarray) -> list[np.ndarray]:
        """Todas as saídas do grafo, numa passada."""
        return self._session.run(None, {self._input: np.ascontiguousarray(batch, np.float32)})

    def __call__(self, batch: np.ndarray, training: bool = False) -> np.ndarray:
        return self.run(batch)[0]

    def dual_forward(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Probabilidades e embedding da mesma passada — o contrato de ``_build_forward``."""
        outputs = self.run(batch)
        return outputs[0], outputs[1] if self.has_embedding else None


def onnx_path(weights_path: Path) -> Path:
    """Onde o ``export_onnx`` grava o grafo de um ``.keras``: ao lado dele."""
    return weights_path.with_suffix(".onnx")


def _shape(dims: list[Any]) -> tuple[int | None, ...]:
    """Dimensões simbólicas (``'batch'``) viram ``None``, como no Keras."""
    return tuple(d if isinstance(d, int) else None for d in dims)
