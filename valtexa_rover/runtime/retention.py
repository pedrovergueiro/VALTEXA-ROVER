"""Política de retenção e os três níveis de dados (brief, seções 13, 16 e 17).

O problema da seção 13: 8 câmeras x 1 quadro/s x 10 h ≈ 864 GB por Rover/dia
em resolução máxima. Enviar tudo é inviável; por isso a decisão acontece aqui,
no robô, **depois** da Edge AI:

* **NORMAL** → só metadados (nível 1) ou imagem reduzida (nível 2), conforme a
  regra; o original é descartado. ``normal_sample_every`` guarda 1 a cada N
  em ORIGINAL para validação (seção 16);
* **POSSÍVEL PRAGA / DOENÇA** → original preservado (nível 3), evento, prioridade
  alta, sequência do buffer circular;
* **ANOMALIA** → original, sequência, evento;
* **baixa confiança** → original, por cima da categoria (nunca rebaixa).

Tudo vem de :class:`~valtexa_rover.config.RetentionConfig` — mudar a política é
mudar o ``rover.json``, não o código.

A limpeza (:meth:`RetentionManager.purge`) apaga do disco o que **já subiu** e
venceu; sob pressão de espaço, apaga o que já subiu mesmo antes de vencer, e
por último as reduzidas NORMAL que ainda não subiram (as de menor prioridade).
Original de evento que não subiu **nunca** é apagado: a seção 12 diz que nada
se perde por falta de conexão — o que se perde, no limite, é o que menos vale.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime, timedelta

from PIL import Image

from valtexa_rover.config import Category, Priority, RetentionConfig, RetentionRule, Tier
from valtexa_rover.runtime.store import LocalStore


@dataclass(frozen=True, slots=True)
class RetentionDecision:
    tier: Tier
    rule: RetentionRule
    priority: Priority
    retention_until: datetime
    preserve_sequence: bool
    sampled_for_validation: bool = False

    @property
    def keeps_image(self) -> bool:
        return self.tier is not Tier.METADADOS


class RetentionManager:
    def __init__(self, config: RetentionConfig) -> None:
        self.config = config
        self._normais = 0

    def decide(
        self, category: Category, low_confidence: bool, captured_at: datetime
    ) -> RetentionDecision:
        regra = self.config.rule_for(category, low_confidence)
        amostra = False
        if category is Category.NORMAL and not low_confidence:
            self._normais += 1
            n = self.config.normal_sample_every
            if n and self._normais % n == 0:
                amostra = True
                regra = RetentionRule(Tier.ORIGINAL, regra.retention_days, regra.priority)
        return RetentionDecision(
            tier=regra.tier,
            rule=regra,
            priority=regra.priority,
            retention_until=captured_at + timedelta(days=regra.retention_days),
            preserve_sequence=regra.preserve_sequence,
            sampled_for_validation=amostra,
        )

    # -- limpeza --------------------------------------------------------------

    def purge(
        self, store: LocalStore, now: datetime, *, free_bytes: int | None = None,
        min_free_bytes: int = 0,
    ) -> dict[str, int]:
        """Apaga o que pode ser apagado. Devolve quantos arquivos e bytes saíram por motivo."""
        saida = {"vencidos": 0, "pressao_enviados": 0, "pressao_normais": 0, "bytes": 0}
        vencidos = store.query(
            "SELECT id, path, size_bytes FROM files WHERE uploaded = 1 AND retention_until < ?",
            (now.isoformat(),),
        )
        saida["bytes"] += self._apaga(store, vencidos)
        saida["vencidos"] = len(vencidos)

        if free_bytes is None or free_bytes >= min_free_bytes:
            return saida
        falta = min_free_bytes - free_bytes - saida["bytes"]
        if falta <= 0:
            return saida
        enviados = store.query(
            "SELECT id, path, size_bytes FROM files WHERE uploaded = 1 ORDER BY created_at"
        )
        escolhidos, falta = _ate_cobrir(enviados, falta)
        saida["bytes"] += self._apaga(store, escolhidos)
        saida["pressao_enviados"] = len(escolhidos)
        if falta <= 0:
            return saida
        # Último recurso: reduzidas de dados NORMAIS que ainda não subiram. O item
        # da fila sai junto — o resultado (metadados) continua e sobe sem imagem.
        normais = store.query(
            "SELECT f.id, f.path, f.size_bytes FROM files f "
            "JOIN sync_queue q ON q.ref_id = f.id AND q.kind = 'file' "
            "WHERE f.uploaded = 0 AND f.tier = ? AND q.priority >= ? AND q.status != 'CONCLUIDO' "
            "ORDER BY f.created_at",
            (Tier.REDUZIDA.value, int(Priority.NORMAL_COMPACTADA)),
        )
        escolhidos, falta = _ate_cobrir(normais, falta)
        with store.transaction() as conn:
            for linha in escolhidos:
                conn.execute(
                    "UPDATE sync_queue SET status = 'FALHOU', last_error = ? WHERE ref_id = ?",
                    ("descartada por falta de espaço (retenção)", linha["id"]),
                )
        saida["bytes"] += self._apaga(store, escolhidos)
        saida["pressao_normais"] = len(escolhidos)
        return saida

    @staticmethod
    def _apaga(store: LocalStore, linhas) -> int:
        total = 0
        for linha in linhas:
            with store.transaction() as conn:
                conn.execute("DELETE FROM files WHERE id = ?", (linha["id"],))
            if store.delete_blob_if_unreferenced(linha["path"]):
                total += int(linha["size_bytes"])
        return total


def _ate_cobrir(linhas, falta: int):
    escolhidos = []
    for linha in linhas:
        if falta <= 0:
            break
        escolhidos.append(linha)
        falta -= int(linha["size_bytes"])
    return escolhidos, falta


def reduce_image(data: bytes, max_side: int, quality: int) -> tuple[bytes, int, int]:
    """JPEG com o lado maior em ``max_side`` (nunca amplia). Devolve bytes e dimensões."""
    with Image.open(io.BytesIO(data)) as image:
        rgb = image.convert("RGB")
        rgb.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
        saida = io.BytesIO()
        rgb.save(saida, format="JPEG", quality=quality, optimize=True)
        return saida.getvalue(), rgb.width, rgb.height
