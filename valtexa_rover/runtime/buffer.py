"""Buffer circular (brief, seção 15): N quadros antes + o do evento + M depois.

Cada câmera tem o seu anel com os últimos ``pre_frames`` quadros, em memória.
Quando a Edge AI marca um evento, o anel vira o "antes" da sequência; os
próximos ``post_frames`` quadros da mesma câmera viram o "depois". Um quadro
mais distante do evento que ``max_duration_s`` não entra (robô parado por muito
tempo não arrasta contexto velho). Se um segundo evento acontece enquanto o
primeiro ainda coleta o "depois", o quadro serve aos dois.

O buffer **só decide quais quadros** formam a sequência. Redimensionar para
``max_side`` (a resolução do contexto), gravar e enfileirar com a prioridade do
buffer é do Core, na mesma transação do quadro que fechou a sequência.

Memória: no máximo ``pre_frames`` quadros por câmera mais os "depois" em aberto.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from valtexa_rover.config import BufferConfig
from valtexa_rover.hal.interfaces import Frame


@dataclass(slots=True)
class BufferedFrame:
    frame: Frame
    result_id: str


@dataclass(slots=True)
class EventSequence:
    event_id: str
    event_record_id: str
    camera_id: str
    event_frame: BufferedFrame
    before: list[BufferedFrame]
    after: list[BufferedFrame] = field(default_factory=list)
    complete: bool = False

    @property
    def context(self) -> list[tuple[str, BufferedFrame]]:
        """Quadros de contexto (sem o do evento), com a posição relativa: ``-2``, ``+1``…"""
        n = len(self.before)
        saida = [(f"{i - n}", q) for i, q in enumerate(self.before)]
        saida += [(f"+{i + 1}", q) for i, q in enumerate(self.after)]
        return saida


class CircularBuffer:
    def __init__(self, config: BufferConfig) -> None:
        self.config = config
        self._rings: dict[str, deque[BufferedFrame]] = {}
        self._open: list[EventSequence] = []

    def observe(
        self, frame: Frame, result_id: str, event: tuple[str, str] | None = None
    ) -> list[EventSequence]:
        """Registra um quadro; ``event=(event_id, record_id)`` se ele disparou um evento.

        Devolve as sequências que este quadro **completou**.
        """
        atual = BufferedFrame(frame, result_id)
        prontas: list[EventSequence] = []
        for seq in self._open:
            if seq.camera_id != frame.camera_id:
                continue
            distancia = (frame.captured_at - seq.event_frame.frame.captured_at).total_seconds()
            if distancia > self.config.max_duration_s:
                seq.complete = True
            else:
                seq.after.append(atual)
                if len(seq.after) >= self.config.post_frames:
                    seq.complete = True
            if seq.complete:
                prontas.append(seq)
        self._open = [s for s in self._open if not s.complete]

        anel = self._rings.setdefault(frame.camera_id, deque(maxlen=max(self.config.pre_frames, 1)))
        if event is not None:
            antes: list[BufferedFrame] = []
            if self.config.pre_frames:
                recentes = [
                    q for q in anel
                    if (frame.captured_at - q.frame.captured_at).total_seconds()
                    <= self.config.max_duration_s
                ]
                antes = recentes[-self.config.pre_frames:]
            seq = EventSequence(event[0], event[1], frame.camera_id, atual, antes)
            if self.config.post_frames == 0:
                seq.complete = True
                prontas.append(seq)
            else:
                self._open.append(seq)
        if self.config.pre_frames:
            anel.append(atual)
        return prontas

    def flush(self) -> list[EventSequence]:
        """Fim da missão: fecha as sequências abertas com o "depois" que tiverem."""
        prontas, self._open = self._open, []
        for seq in prontas:
            seq.complete = True
        return prontas

    @property
    def open_sequences(self) -> int:
        return len(self._open)
