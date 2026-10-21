"""Teste retirado de tests/test_video.py em 20/10/2026: prioridade do vídeo no rádio da fazenda."""

from valtexa_rover.enlace.rpc import P3, PRAZO_VIDEO, classifica


def test_enlace_video_e_slot_so_o_ultimo_abaixo_de_comando() -> None:
    prio, slot = classifica("POST", "/v1/device/video/cam-2", b"")
    assert prio == P3 and slot == "video:cam-2"
    assert classifica("GET", "/v1/device/commands", b"")[0] < P3
    assert PRAZO_VIDEO <= 3.0
