"""Relógio injetável.

O Runtime nunca chama ``datetime.now`` nem ``time.sleep`` direto: pede ao
relógio. Em campo é o do sistema (:class:`SystemClock`); nos testes e no
simulador é :class:`ManualClock`, que só anda quando mandam — é o que deixa
testar *backoff* de minutos e missões de horas em milissegundos.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Protocol

from valtexa_rover._compat import UTC


class Clock(Protocol):
    def now(self) -> datetime: ...

    def monotonic(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(max(0.0, seconds))


class ManualClock:
    """Relógio de teste/simulação: ``advance`` e ``sleep`` movem o tempo na hora."""

    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 10, 4, 12, 0, tzinfo=UTC)
        self._mono = 0.0

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._mono

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)
        self._mono += seconds

    def sleep(self, seconds: float) -> None:
        self.advance(max(0.0, seconds))
