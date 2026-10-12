"""Perfil do modelo na Edge AI: como recortar o quadro e quando ele vira evento.

Viaja com o modelo, ao lado do ``.onnx`` (``<modelo>.edge.json``), e não no
``rover.json``: os limiares foram medidos **para aquele modelo** e mudam junto
com ele. É gerado por ``ia/training/fit_edge.py`` com plantas rotuladas.

Os modelos foliares da ValTexa foram treinados com uma folha (ou um trecho de
caule) preenchendo 224 px — 15 a 25 cm de planta. O quadro do Rover tem ~1 m de
planta: reduzido inteiro a 224 px, uma lesão some. Então o quadro é cortado em
recortes de **tamanho físico fixo** (``tile_mm``), pela distância até a planta (a
profundidade da OAK-D) e pelo campo de visão, com a geometria copiada do ``ia/``
(:mod:`valtexa_rover.edge_ai.vendor.rover_geometry`).

Decisão do quadro (medida, não escolhida):

* com ``evidence`` (o caso dos modelos reais): cada recorte vira a log-razão de
  verossimilhança da faixa da sua probabilidade, por doença (tabelas medidas em plantas
  rotuladas, limitadas a ``±llr_clip``); o escore do quadro é a maior, entre as doenças,
  da média nos recortes vezes o tamanho efetivo ``n / (1 + (n - 1) rho)``. Evento:
  escore > ``event_evidence``. É a mesma evidência da fusão por planta da nuvem, num
  quadro só — a contagem de recortes não cabe no orçamento, porque o modelo põe
  probabilidade ≥ 0,99 de doença em pedaço de folha sadia (medido em 07/10/2026);
* sem ``evidence``: evento quando pelo menos ``min_event_tiles`` recortes têm
  probabilidade de doença/praga ≥ ``event_threshold``.

Sem perfil, o motor classifica o quadro inteiro (modelo de demonstração).

**Cabeça sobre a torre de visão** (``head``; o caso do café): a classificação de cada
recorte não é da CNN, é da cabeça (``<modelo>.head.npz``) sobre o embedding da torre de
visão do SigLIP 2 (``<modelo>.encoder.onnx``, conferida pelo SHA-256 do perfil). No café
a CNN sozinha acerta 6,5% num acervo externo; com a cabeça, 80,1%.
"""

from __future__ import annotations

import bisect
import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROFILE_VERSION = 1


class ProfileError(ValueError):
    """``edge.json`` ausente de campo, incoerente ou de outra versão."""


@dataclass(frozen=True, slots=True)
class EvidenceRule:
    """Evidência por quadro (``training.fit_edge``): tabelas de LLR por faixa e limiares."""

    bin_edges: tuple[float, ...]
    llr_tables: dict[str, tuple[float, ...]]
    llr_clip: float
    tile_correlation: float
    event_evidence: float
    near_miss_evidence: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceRule:
        conhecidas = set(cls.__dataclass_fields__)
        sobra = set(data) - conhecidas
        if sobra:
            raise ProfileError(f"evidence: chave(s) desconhecida(s): {sorted(sobra)}")
        try:
            regra = cls(
                bin_edges=tuple(float(v) for v in data["bin_edges"]),
                llr_tables={str(d): tuple(float(v) for v in t)
                            for d, t in data["llr_tables"].items()},
                llr_clip=float(data["llr_clip"]),
                tile_correlation=float(data["tile_correlation"]),
                event_evidence=float(data["event_evidence"]),
                near_miss_evidence=(None if data.get("near_miss_evidence") is None
                                    else float(data["near_miss_evidence"])),
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ProfileError(f"evidence: {exc}") from exc
        return regra.validate()

    def validate(self) -> EvidenceRule:
        e = self.bin_edges
        crescente = all(b > a for a, b in itertools.pairwise(e))
        if len(e) < 2 or e[0] != 0.0 or e[-1] != 1.0 or not crescente:
            raise ProfileError("evidence.bin_edges: de 0 a 1, estritamente crescente.")
        if not self.llr_tables:
            raise ProfileError("evidence.llr_tables vazio.")
        for doenca, tabela in self.llr_tables.items():
            if len(tabela) != len(e) - 1 or not all(math.isfinite(v) for v in tabela):
                raise ProfileError(f"evidence.llr_tables[{doenca!r}]: {len(e) - 1} faixas finitas.")
        if not (self.llr_clip > 0 and 0.0 <= self.tile_correlation <= 1.0):
            raise ProfileError("evidence: llr_clip > 0 e tile_correlation em [0, 1].")
        if not math.isfinite(self.event_evidence):
            raise ProfileError("evidence.event_evidence não finito.")
        if self.near_miss_evidence is not None and self.near_miss_evidence > self.event_evidence:
            raise ProfileError("evidence.near_miss_evidence não pode passar de event_evidence.")
        return self

    def tile_llr(self, doenca: str, p: float) -> float:
        i = min(max(bisect.bisect_right(self.bin_edges, p) - 1, 0), len(self.bin_edges) - 2)
        return min(max(self.llr_tables[doenca][i], -self.llr_clip), self.llr_clip)

    def score(self, doenca: str, probs: list[float]) -> float:
        """Média da LLR nos recortes vezes o tamanho efetivo (desconto de correlação)."""
        n = len(probs)
        if n == 0:
            return 0.0
        media = sum(self.tile_llr(doenca, p) for p in probs) / n
        return media * n / (1.0 + (n - 1) * self.tile_correlation)


@dataclass(frozen=True, slots=True)
class HeadSpec:
    """A torre de visão que a cabeça espera (``training.fit_edge``, bloco ``head``)."""

    encoder_model_id: str
    encoder_sha256: str
    input_size: int = 224

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HeadSpec:
        sobra = set(data) - set(cls.__dataclass_fields__)
        if sobra:
            raise ProfileError(f"head: chave(s) desconhecida(s): {sorted(sobra)}")
        try:
            spec = cls(str(data["encoder_model_id"]), str(data["encoder_sha256"]).lower(),
                       int(data.get("input_size", 224)))
        except (KeyError, TypeError, ValueError) as exc:
            raise ProfileError(f"head: {exc}") from exc
        if len(spec.encoder_sha256) != 64 or not all(
                c in "0123456789abcdef" for c in spec.encoder_sha256):
            raise ProfileError("head.encoder_sha256 deveria ser um SHA-256 em hexadecimal.")
        if spec.input_size < 32:
            raise ProfileError("head.input_size inválido.")
        return spec


@dataclass(frozen=True, slots=True)
class EdgeProfile:
    crop: str
    healthy_label: str
    event_threshold: float
    """Probabilidade de doença/praga num recorte a partir da qual ele conta como achado."""
    min_event_tiles: int = 1
    """Quantos recortes achados fazem do quadro um evento."""
    near_miss_threshold: float | None = None
    """Quadro abaixo do evento mas com recorte acima disto → ``baixa_confianca`` (validação)."""
    anomaly_mean_confidence_below: float | None = None
    """Média, nos recortes, da maior probabilidade abaixo disto → ANOMALIA (o modelo não
    reconhece o que vê). ``None`` desliga."""
    tile_mm: float = 150.0
    overlap: float = 0.25
    hfov_deg: float = 127.0
    vfov_deg: float = 110.0
    default_distance_mm: float = 270.0
    """Distância usada quando a câmera não mede profundidade no quadro."""
    tile_batch: int = 32
    tile_probabilities: str = "full"
    """O que dos recortes vai nos metadados: ``full`` (vetor inteiro de cada recorte —
    a nuvem funde por planta com isso), ``top`` (só a melhor classe) ou ``none``."""
    evidence: EvidenceRule | None = None
    """Regra por evidência (preferida). Sem ela, a contagem de recortes."""
    head: HeadSpec | None = None
    """Classificar cada recorte pela cabeça sobre a torre de visão (café)."""
    measured: dict[str, Any] = field(default_factory=dict)
    """O que a calibração mediu (sensibilidade, alarme falso, amostra). Só registro."""

    def validate(self) -> EdgeProfile:
        for nome, valor in (("event_threshold", self.event_threshold),
                            ("near_miss_threshold", self.near_miss_threshold),
                            ("anomaly_mean_confidence_below", self.anomaly_mean_confidence_below)):
            if valor is not None and not (0.0 <= valor <= 1.0 and math.isfinite(valor)):
                raise ProfileError(f"{nome} fora de [0, 1]: {valor}")
        quase = self.near_miss_threshold
        if quase is not None and quase > self.event_threshold:
            raise ProfileError("near_miss_threshold não pode passar de event_threshold.")
        if self.min_event_tiles < 1:
            raise ProfileError("min_event_tiles precisa ser ≥ 1.")
        geometria_ok = (self.tile_mm > 0 and 0.0 <= self.overlap < 1.0
                        and self.default_distance_mm > 0)
        if not geometria_ok:
            raise ProfileError("tile_mm/overlap/default_distance_mm inválidos.")
        if not (0 < self.hfov_deg < 180 and 0 < self.vfov_deg < 180):
            raise ProfileError("campo de visão inválido.")
        if self.tile_probabilities not in ("full", "top", "none"):
            raise ProfileError("tile_probabilities: use full, top ou none.")
        if self.tile_batch < 1:
            raise ProfileError("tile_batch precisa ser ≥ 1.")
        return self

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EdgeProfile:
        if data.get("version") != PROFILE_VERSION:
            raise ProfileError(f"versão do perfil não suportada: {data.get('version')!r}")
        conhecidas = set(cls.__dataclass_fields__) | {"version"}
        sobra = set(data) - conhecidas
        if sobra:
            raise ProfileError(f"chave(s) desconhecida(s): {sorted(sobra)}")
        valores = {k: v for k, v in data.items() if k != "version"}
        if valores.get("evidence") is not None:
            if not isinstance(valores["evidence"], dict):
                raise ProfileError("evidence deveria ser um objeto.")
            valores["evidence"] = EvidenceRule.from_dict(valores["evidence"])
        if valores.get("head") is not None:
            if not isinstance(valores["head"], dict):
                raise ProfileError("head deveria ser um objeto.")
            valores["head"] = HeadSpec.from_dict(valores["head"])
        try:
            return cls(**valores).validate()
        except TypeError as exc:
            raise ProfileError(str(exc)) from exc

    @classmethod
    def load(cls, path: Path) -> EdgeProfile:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ProfileError(f"{path.name}: JSON inválido ({exc})") from exc
        return cls.from_dict(data)
