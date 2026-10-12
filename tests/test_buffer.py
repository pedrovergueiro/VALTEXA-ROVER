from __future__ import annotations

from datetime import datetime, timedelta

from valtexa_rover._compat import UTC
from valtexa_rover.config import BufferConfig
from valtexa_rover.hal.interfaces import Frame
from valtexa_rover.runtime.buffer import CircularBuffer

T0 = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)


def quadro(i: int, cam: str = "cam-0", t: float | None = None) -> Frame:
    return Frame(cam, i, T0 + timedelta(seconds=i if t is None else t), b"x", "image/jpeg", 1, 1)


def test_antes_evento_depois() -> None:
    buf = CircularBuffer(BufferConfig(pre_frames=3, post_frames=2, max_duration_s=60))
    for i in range(5):
        assert buf.observe(quadro(i), f"r{i}") == []
    assert buf.observe(quadro(5), "r5", ("ev", "r5")) == []
    assert buf.observe(quadro(6), "r6") == []
    prontas = buf.observe(quadro(7), "r7")
    assert len(prontas) == 1
    seq = prontas[0]
    assert [q.result_id for q in seq.before] == ["r2", "r3", "r4"]
    assert seq.event_frame.result_id == "r5"
    assert [q.result_id for q in seq.after] == ["r6", "r7"]
    assert [p for p, _ in seq.context] == ["-3", "-2", "-1", "+1", "+2"]


def test_duracao_maxima_corta_contexto_velho() -> None:
    buf = CircularBuffer(BufferConfig(pre_frames=3, post_frames=3, max_duration_s=5))
    buf.observe(quadro(0, t=0), "velho")
    buf.observe(quadro(1, t=10), "r1")
    buf.observe(quadro(2, t=12), "ev", ("ev", "ev"))
    prontas = buf.observe(quadro(3, t=30), "tarde")  # 18 s depois: fecha sem ele
    assert len(prontas) == 1
    assert [q.result_id for q in prontas[0].before] == ["r1"]
    assert prontas[0].after == []


def test_eventos_sobrepostos_compartilham_quadros_e_cameras_separadas() -> None:
    buf = CircularBuffer(BufferConfig(pre_frames=1, post_frames=2, max_duration_s=60))
    buf.observe(quadro(0), "r0")
    buf.observe(quadro(1), "e1", ("ev1", "e1"))
    buf.observe(quadro(1, cam="cam-1"), "outra")  # outra câmera não entra
    buf.observe(quadro(2), "e2", ("ev2", "e2"))
    prontas = buf.observe(quadro(3), "r3")
    assert [s.event_id for s in prontas] == ["ev1"]
    assert [q.result_id for q in prontas[0].after] == ["e2", "r3"]
    assert buf.open_sequences == 1
    final = buf.flush()
    assert [s.event_id for s in final] == ["ev2"]
    assert [q.result_id for q in final[0].before] == ["e1"]
    assert [q.result_id for q in final[0].after] == ["r3"]


def test_sem_depois_fecha_na_hora() -> None:
    buf = CircularBuffer(BufferConfig(pre_frames=2, post_frames=0))
    buf.observe(quadro(0), "r0")
    prontas = buf.observe(quadro(1), "ev", ("ev", "ev"))
    assert len(prontas) == 1 and [q.result_id for q in prontas[0].before] == ["r0"]
