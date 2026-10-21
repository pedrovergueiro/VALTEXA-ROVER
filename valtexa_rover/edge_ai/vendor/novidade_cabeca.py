"""Novidade no embedding da torre de visão: a cabeça reconhece quando não sabe.

O softmax da cabeça sempre escolhe uma das classes que aprendeu. Diante de algo que nunca
viu — folha de outra planta no vídeo, a lavoura sob outra luz, uma praga fora do catálogo — ele
pode dar 95% de ferrugem com a mesma convicção de um caso de livro (medido em vídeo real,
16/10/2026: 89% dos quadros que passavam no portão saíam ``Rust``, xícara de café inclusive).
Confiança alta não é informação suficiente.

O :mod:`app.inference.novelty` responde isso para a CNN (embedding da penúltima camada); no
café, quem decide é a cabeça sobre o embedding do SigLIP 2, e a CNN nem roda no Rover. Aqui é a
mesma pergunta no espaço certo:

1. o embedding cru é normalizado (L2), como a cabeça faz, e projetado numa base de PCA
   ajustada no treino (``k`` componentes: em 768 dimensões, com poucas centenas de folhas por
   classe, a covariância cheia não se estima);
2. por classe, distância de Mahalanobis ao centro, com a covariância **da classe** encolhida
   para a diagonal (como no ``novelty``: covariância comum incha e esconde o estranho);
3. o limiar de cada classe é o percentil ``p`` das distâncias num conjunto de **calibração**
   que não entrou no ajuste — no treino as distâncias são otimistas e o alarme em campo
   sairia maior que o prometido;
4. o escore é a menor razão distância/limiar entre as classes: ``> 1`` = longe de tudo o que
   a cabeça conhece.

É um orçamento do tempo do agrônomo, não uma garantia: com ``p = 99`` cerca de 1% do que é do
domínio também sai marcado. Só biblioteca padrão e NumPy: o Rover usa este arquivo sem o resto
do ``ia/`` (cópia byte a byte em ``valtexa_rover/edge_ai/vendor/``). Arquivo ``.npz`` sem
pickle.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

FORMATO = 1
COMPONENTES = 64
PERCENTIL = 99.0
ENCOLHIMENTO = 0.1


def _l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


@dataclass(frozen=True)
class ReferenciaDeNovidade:
    encoder: str
    """``model_id`` da torre: embedding de outra torre é outro espaço."""
    class_names: tuple[str, ...]
    media: np.ndarray
    """``(D,)``: centro dos embeddings normalizados do ajuste."""
    componentes: np.ndarray
    """``(k, D)``: a base da PCA."""
    centroides: np.ndarray
    """``(C, k)``."""
    precisoes: np.ndarray
    """``(C, k, k)``: a inversa da covariância encolhida de cada classe."""
    limiares: np.ndarray
    """``(C,)``: o percentil das distâncias da calibração, por classe."""
    percentil: float = PERCENTIL
    medido: dict[str, Any] = field(default_factory=dict)
    """Quantos de cada classe no ajuste e na calibração, e a taxa marcada na calibração."""

    def projeta(self, embeddings: np.ndarray) -> np.ndarray:
        x = _l2(np.atleast_2d(embeddings))
        if x.shape[1] != self.media.shape[0]:
            raise ValueError(f"embedding de {x.shape[1]} dimensões; a referência espera "
                             f"{self.media.shape[0]}.")
        return (x - self.media) @ self.componentes.T

    def distancias(self, embeddings: np.ndarray) -> np.ndarray:
        """``(N, C)``: Mahalanobis de cada embedding ao centro de cada classe."""
        z = self.projeta(embeddings)
        d = z[:, None, :] - self.centroides[None, :, :]
        return np.sqrt(np.maximum(np.einsum("nck,ckl,ncl->nc", d, self.precisoes, d), 0.0))

    def escore(self, embeddings: np.ndarray) -> np.ndarray:
        """``(N,)``: a menor razão distância/limiar. ``> 1`` = novidade."""
        return (self.distancias(embeddings) / self.limiares[None, :]).min(axis=1)

    def novos(self, embeddings: np.ndarray) -> np.ndarray:
        return self.escore(embeddings) > 1.0

    # -- arquivo --------------------------------------------------------------------------

    def save(self, caminho: Path | str) -> None:
        meta = {"formato": FORMATO, "encoder": self.encoder,
                "class_names": list(self.class_names), "percentil": self.percentil,
                "medido": self.medido}
        np.savez_compressed(caminho, media=self.media.astype(np.float32),
                            componentes=self.componentes.astype(np.float32),
                            centroides=self.centroides.astype(np.float32),
                            precisoes=self.precisoes.astype(np.float32),
                            limiares=self.limiares.astype(np.float64),
                            meta=np.array(json.dumps(meta, ensure_ascii=False)))

    @classmethod
    def load(cls, caminho: Path | str) -> ReferenciaDeNovidade:
        try:
            with np.load(caminho, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"]))
                partes = {k: np.asarray(z[k], dtype=np.float64)
                          for k in ("media", "componentes", "centroides", "precisoes",
                                    "limiares")}
        except (OSError, KeyError, ValueError) as exc:
            raise ValueError(f"{Path(caminho).name}: referência de novidade ilegível ({exc}).") \
                from exc
        if meta.get("formato") != FORMATO:
            raise ValueError(f"{Path(caminho).name}: formato {meta.get('formato')!r} desconhecido.")
        ref = cls(encoder=str(meta["encoder"]), class_names=tuple(meta["class_names"]),
                  percentil=float(meta["percentil"]), medido=dict(meta.get("medido") or {}),
                  **partes)
        c, k = ref.centroides.shape
        if (len(ref.class_names) != c or ref.componentes.shape[0] != k
                or ref.precisoes.shape != (c, k, k) or ref.limiares.shape != (c,)
                or not np.all(ref.limiares > 0)):
            raise ValueError(f"{Path(caminho).name}: formas incoerentes.")
        return ref


def _precisao(desvios: np.ndarray, encolhimento: float) -> np.ndarray:
    cov = np.cov(desvios, rowvar=False) if len(desvios) > 1 else np.eye(desvios.shape[1])
    cov = np.atleast_2d(cov)
    alvo = np.diag(np.maximum(np.diag(cov), 1e-9))
    cov = (1.0 - encolhimento) * cov + encolhimento * alvo
    return np.linalg.pinv(cov, hermitian=True)


def ajusta(emb_ajuste: np.ndarray, y_ajuste: np.ndarray, emb_calib: np.ndarray,
           y_calib: np.ndarray, class_names: Sequence[str], encoder: str, *,
           componentes: int = COMPONENTES, percentil: float = PERCENTIL,
           encolhimento: float = ENCOLHIMENTO) -> ReferenciaDeNovidade:
    """A referência: PCA e covariâncias no ``ajuste``, limiares na ``calibração``."""
    x = _l2(emb_ajuste)
    y = np.asarray(y_ajuste, dtype=np.int64)
    c = len(class_names)
    if len(x) != len(y) or len(_l2(emb_calib)) != len(y_calib):
        raise ValueError("embeddings e rótulos com tamanhos diferentes.")
    faltam = [class_names[k] for k in range(c) if (y == k).sum() < 2
              or (np.asarray(y_calib) == k).sum() < 5]
    if faltam:
        raise ValueError(f"classes sem exemplos bastantes para ajustar e calibrar: {faltam}")
    media = x.mean(axis=0)
    k = int(min(componentes, x.shape[1], len(x) - 1))
    _u, _s, vt = np.linalg.svd(x - media, full_matrices=False)
    base = vt[:k]
    z = (x - media) @ base.T
    centroides = np.stack([z[y == j].mean(axis=0) for j in range(c)])
    precisoes = np.stack([_precisao(z[y == j] - centroides[j], encolhimento) for j in range(c)])
    provisoria = ReferenciaDeNovidade(encoder, tuple(class_names), media, base, centroides,
                                      precisoes, np.ones(c), percentil)
    yc = np.asarray(y_calib, dtype=np.int64)
    dc = provisoria.distancias(emb_calib)[np.arange(len(yc)), yc]
    limiares = np.array([np.percentile(dc[yc == j], percentil) for j in range(c)])
    ref = ReferenciaDeNovidade(encoder, tuple(class_names), media, base, centroides, precisoes,
                               limiares, percentil)
    marcados = float(ref.novos(emb_calib).mean())
    medido = {"componentes": k, "encolhimento": encolhimento,
              "ajuste": {class_names[j]: int((y == j).sum()) for j in range(c)},
              "calibracao": {class_names[j]: int((yc == j).sum()) for j in range(c)},
              "marcados_na_calibracao": round(marcados, 4)}
    return ReferenciaDeNovidade(encoder, tuple(class_names), media, base, centroides, precisoes,
                                limiares, percentil, medido)
