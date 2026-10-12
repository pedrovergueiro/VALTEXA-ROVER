"""Configuration Manager: tudo que muda o comportamento do Rover sem mudar código.

A configuração é um JSON (``rover.json``) com seções; o que não vier fica no
padrão daqui. Chave desconhecida é **erro** (:class:`ConfigError`), não
silêncio: um ``"retencao"`` digitado errado não pode deixar o robô guardando
tudo em resolução máxima sem ninguém perceber.

As seções:

* ``classificacao`` — limiares de baixa confiança e de anomalia (seção 14);
* ``retencao`` — o que fica de cada categoria: só metadados, imagem reduzida ou
  original, por quantos dias e com que prioridade de upload (seções 16, 17, 19);
* ``buffer`` — quadros antes/depois de um evento, duração, resolução (seção 15);
* ``sync`` — chunk, lote, *backoff* (seção 18);
* ``saude`` — limites do Health Monitor.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from enum import IntEnum
from pathlib import Path
from typing import Any

from valtexa_rover._compat import StrEnum


class ConfigError(ValueError):
    """Configuração inválida. A mensagem diz a chave e o problema."""


class Category(StrEnum):
    """Categorias da Edge AI (brief, seção 14)."""

    NORMAL = "NORMAL"
    POSSIVEL_PRAGA = "POSSIVEL_PRAGA"
    POSSIVEL_DOENCA = "POSSIVEL_DOENCA"
    ANOMALIA = "ANOMALIA"


class Tier(StrEnum):
    """Os três níveis de dados (brief, seção 16)."""

    METADADOS = "METADADOS"
    REDUZIDA = "REDUZIDA"
    ORIGINAL = "ORIGINAL"


class Priority(IntEnum):
    """Prioridade de upload (brief, seção 19). Menor número sobe primeiro."""

    EVENTO_CRITICO = 1
    DOENCA_PRAGA = 2
    ANOMALIA = 3
    CONCLUSAO_MISSAO = 4
    NORMAL_COMPACTADA = 5
    SECUNDARIO = 6


@dataclass(frozen=True, slots=True)
class ClassificationConfig:
    low_confidence_below: float = 0.60
    """Abaixo disto o resultado é marcado ``baixa_confianca`` (e preserva o original)."""
    anomaly_below: float = 0.30
    """Abaixo disto nenhuma classe foi reconhecida: vira ANOMALIA (fora do que o modelo conhece)."""

    def validate(self) -> None:
        if not 0 <= self.anomaly_below <= self.low_confidence_below <= 1:
            raise ConfigError(
                "classificacao: exige 0 ≤ anomaly_below ≤ low_confidence_below ≤ 1."
            )


@dataclass(frozen=True, slots=True)
class RetentionRule:
    tier: Tier
    retention_days: int
    priority: Priority
    max_side: int = 0
    """Lado maior da imagem reduzida, em pixels (só vale para ``REDUZIDA``)."""
    jpeg_quality: int = 70
    preserve_sequence: bool = False
    """Guardar também a sequência do buffer circular (quadros antes/depois)."""

    def validate(self, nome: str) -> None:
        if self.retention_days < 0:
            raise ConfigError(f"retencao.{nome}: retention_days negativo.")
        if self.tier is Tier.REDUZIDA and self.max_side < 16:
            raise ConfigError(f"retencao.{nome}: REDUZIDA exige max_side ≥ 16.")
        if not 1 <= self.jpeg_quality <= 95:
            raise ConfigError(f"retencao.{nome}: jpeg_quality fora de 1 a 95.")


def _regras_padrao() -> dict[str, RetentionRule]:
    return {
        Category.NORMAL: RetentionRule(Tier.REDUZIDA, 7, Priority.NORMAL_COMPACTADA, 320, 60),
        Category.POSSIVEL_PRAGA: RetentionRule(
            Tier.ORIGINAL, 90, Priority.DOENCA_PRAGA, preserve_sequence=True
        ),
        Category.POSSIVEL_DOENCA: RetentionRule(
            Tier.ORIGINAL, 90, Priority.DOENCA_PRAGA, preserve_sequence=True
        ),
        Category.ANOMALIA: RetentionRule(
            Tier.ORIGINAL, 90, Priority.ANOMALIA, preserve_sequence=True
        ),
        "BAIXA_CONFIANCA": RetentionRule(Tier.ORIGINAL, 30, Priority.ANOMALIA),
    }


@dataclass(frozen=True, slots=True)
class RetentionConfig:
    """Política de retenção. ``BAIXA_CONFIANCA`` vale por cima da categoria."""

    rules: dict[str, RetentionRule] = field(default_factory=_regras_padrao)
    normal_sample_every: int = 0
    """Uma a cada N imagens NORMAL vai em ORIGINAL, para validação (0 = nunca)."""

    def rule_for(self, category: Category, low_confidence: bool) -> RetentionRule:
        if low_confidence and "BAIXA_CONFIANCA" in self.rules:
            regra = self.rules["BAIXA_CONFIANCA"]
            base = self.rules[category]
            # Baixa confiança nunca *rebaixa*: uma DOENÇA duvidosa continua com a
            # prioridade e a sequência de doença.
            return replace(
                regra,
                priority=min(regra.priority, base.priority),
                retention_days=max(regra.retention_days, base.retention_days),
                preserve_sequence=regra.preserve_sequence or base.preserve_sequence,
            )
        return self.rules[category]

    def validate(self) -> None:
        for categoria in Category:
            if categoria not in self.rules:
                raise ConfigError(f"retencao: falta a regra de {categoria.value}.")
        for nome, regra in self.rules.items():
            regra.validate(str(nome))
        if self.normal_sample_every < 0:
            raise ConfigError("retencao.normal_sample_every negativo.")


@dataclass(frozen=True, slots=True)
class BufferConfig:
    pre_frames: int = 3
    post_frames: int = 3
    max_duration_s: float = 10.0
    """Quadro mais velho que isto (em relação ao evento) não entra na sequência."""
    max_side: int = 1024
    """Resolução dos quadros de contexto guardados (o do evento fica original)."""
    jpeg_quality: int = 85
    priority: Priority = Priority.ANOMALIA

    def validate(self) -> None:
        if self.pre_frames < 0 or self.post_frames < 0:
            raise ConfigError("buffer: pre_frames e post_frames não podem ser negativos.")
        if self.max_duration_s <= 0:
            raise ConfigError("buffer.max_duration_s precisa ser positivo.")
        if self.max_side < 16:
            raise ConfigError("buffer.max_side precisa ser ≥ 16.")


@dataclass(frozen=True, slots=True)
class SyncConfig:
    chunk_size: int = 256 * 1024
    batch_size: int = 200
    """Registros (resultados/eventos/telemetria) por requisição."""
    backoff_base_s: float = 1.0
    backoff_max_s: float = 60.0
    max_attempts: int = 0
    """Tentativas antes de desistir de um item por erro do servidor (0 = nunca desiste)."""

    def validate(self) -> None:
        if self.chunk_size < 1024:
            raise ConfigError("sync.chunk_size precisa ser ≥ 1024 bytes.")
        if self.batch_size < 1:
            raise ConfigError("sync.batch_size precisa ser ≥ 1.")
        if self.backoff_base_s <= 0 or self.backoff_max_s < self.backoff_base_s:
            raise ConfigError("sync: exige 0 < backoff_base_s ≤ backoff_max_s.")


@dataclass(frozen=True, slots=True)
class HealthConfig:
    min_free_bytes: int = 2 * 1024**3
    """Abaixo disto: falha ``storage_low`` e a retenção apaga o que já subiu e venceu."""
    max_temperature_c: float = 70.0

    def validate(self) -> None:
        if self.min_free_bytes < 0:
            raise ConfigError("saude.min_free_bytes negativo.")


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    classificacao: ClassificationConfig = field(default_factory=ClassificationConfig)
    retencao: RetentionConfig = field(default_factory=RetentionConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    saude: HealthConfig = field(default_factory=HealthConfig)

    def validate(self) -> RuntimeConfig:
        self.classificacao.validate()
        self.retencao.validate()
        self.buffer.validate()
        self.sync.validate()
        self.saude.validate()
        return self

    # -- JSON ---------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RuntimeConfig:
        _sem_desconhecidas("raiz", data, {f.name for f in fields(cls)})
        return cls(
            classificacao=_dc(ClassificationConfig, data.get("classificacao"), "classificacao"),
            retencao=_retencao(data.get("retencao")),
            buffer=_dc(BufferConfig, data.get("buffer"), "buffer", {"priority": Priority}),
            sync=_dc(SyncConfig, data.get("sync"), "sync"),
            saude=_dc(HealthConfig, data.get("saude"), "saude"),
        ).validate()

    @classmethod
    def load(cls, path: Path | str | None) -> RuntimeConfig:
        if path is None or not Path(path).exists():
            return cls().validate()
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{path}: JSON inválido ({exc}).") from exc
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        bruto = asdict(self)
        bruto["retencao"]["rules"] = {
            str(k): {**asdict(v), "tier": v.tier.value, "priority": int(v.priority)}
            for k, v in self.retencao.rules.items()
        }
        bruto["buffer"]["priority"] = int(self.buffer.priority)
        return bruto


def _sem_desconhecidas(secao: str, data: dict, conhecidas: set[str]) -> None:
    sobra = set(data) - conhecidas
    if sobra:
        raise ConfigError(f"{secao}: chave(s) desconhecida(s) {sorted(sobra)}.")


def _dc(tipo, data: dict | None, secao: str, conversores: dict | None = None):
    if data is None:
        return tipo()
    if not isinstance(data, dict):
        raise ConfigError(f"{secao}: esperava um objeto.")
    _sem_desconhecidas(secao, data, {f.name for f in fields(tipo)})
    valores = dict(data)
    for chave, conv in (conversores or {}).items():
        if chave in valores:
            valores[chave] = conv(valores[chave])
    try:
        return tipo(**valores)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{secao}: {exc}") from exc


def _retencao(data: dict | None) -> RetentionConfig:
    if data is None:
        return RetentionConfig()
    _sem_desconhecidas("retencao", data, {"rules", "normal_sample_every"})
    regras = _regras_padrao()
    nomes_validos = {c.value for c in Category} | {"BAIXA_CONFIANCA"}
    for nome, bruto in (data.get("rules") or {}).items():
        if nome not in nomes_validos:
            raise ConfigError(f"retencao.rules: categoria desconhecida {nome!r}.")
        base = regras[Category(nome) if nome != "BAIXA_CONFIANCA" else nome]
        _sem_desconhecidas(f"retencao.rules.{nome}", bruto, {f.name for f in fields(RetentionRule)})
        try:
            valores = {**asdict(base), **bruto}
            valores["tier"] = Tier(valores["tier"])
            valores["priority"] = Priority(valores["priority"])
        except ValueError as exc:
            raise ConfigError(f"retencao.rules.{nome}: {exc}") from exc
        chave = Category(nome) if nome != "BAIXA_CONFIANCA" else nome
        regras[chave] = RetentionRule(**valores)
    amostra = int(data.get("normal_sample_every", 0))
    return RetentionConfig(rules=regras, normal_sample_every=amostra)
