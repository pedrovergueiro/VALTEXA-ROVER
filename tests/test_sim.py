"""Simulador do Rover (seção 41): o Runtime de verdade sobre a HAL simulada.

A nuvem é o DUBLÊ do conftest; o simulador contra o SaaS real está em
``saas/tests/test_simulador_unico.py``.
"""

from __future__ import annotations

import json

from conftest import FakeCloud, designation

from valtexa_rover.runtime.client import DeviceIdentity, generate_keypair
from valtexa_rover.sim import MODELO_DEMO_SHA256, CampoSimulado, PlanoDeRede, RoverSimulado


def _rover(tmp_path, clock, **kwargs):
    privada, publica = generate_keypair()
    nuvem = FakeCloud(publica, [designation("talhao-1", MODELO_DEMO_SHA256)])
    campo = CampoSimulado.gerar(-47.06, -22.90, linhas=2, plantas_por_linha=15,
                                doentes={"L1-P005": "doenca", "L2-P010": "praga"})
    rover = RoverSimulado(tmp_path / "rover", DeviceIdentity("dev-1", "dk", privada), nuvem,
                          campo, clock=clock, cameras=2, auto_sync=True, **kwargs)
    return rover, nuvem, campo


def test_modelo_demo_e_o_que_o_hash_diz() -> None:
    import hashlib

    from valtexa_rover.sim import MODELO_DEMO
    assert hashlib.sha256(MODELO_DEMO.read_bytes()).hexdigest() == MODELO_DEMO_SHA256


def test_missao_simulada_com_queda_reinicio_e_camera_fora(tmp_path, clock) -> None:
    rover, nuvem, _campo = _rover(
        tmp_path, clock, camera_indisponivel=lambda cid, n: cid == "cam-1" and 10 <= n < 14)
    rover.core.start_mission("m-sim", "talhao-1")
    plano = PlanoDeRede(sem_rede=[(8, 30)])
    r1 = rover.rodar(20, plano)
    rover.reiniciar()  # no meio da janela sem rede
    r2 = rover.rodar(25, plano, inicio=20)
    rover.core.complete_mission()
    assert rover.sincronizar() == 0
    quadros = r1.quadros + r2.quadros
    assert quadros == 45 * 2 - (r1.falhas_de_camera + r2.falhas_de_camera)
    assert r1.falhas_de_camera + r2.falhas_de_camera == 4
    assert r1.passos_offline + r2.passos_offline == 22
    # Tudo que o robô produziu chegou à nuvem uma vez só.
    locais = rover.core.store.scalar("SELECT COUNT(*) FROM records")
    assert len(nuvem.records) == locais
    resultados = [b for b in nuvem.record_bodies.values() if b["kind"] == "result"]
    assert len(resultados) == quadros
    # Os eventos aconteceram diante das plantas doentes, e o resultado diz qual planta.
    eventos = [b for b in resultados if b["type"] in ("POSSIVEL_DOENCA", "POSSIVEL_PRAGA")]
    assert eventos, "o rover passou pelas plantas doentes"
    plantas = {b["region"]["plant_id"] for b in eventos if b["region"]}
    assert plantas <= {"L1-P004", "L1-P005", "L1-P006", "L2-P009", "L2-P010", "L2-P011"}
    assert all(b["hardware"]["simulated"] for b in resultados)
    tipos = {b["type"] for b in nuvem.record_bodies.values()}
    assert {"MISSION_STARTED", "MISSION_RESUMED", "MISSION_COMPLETED", "HARDWARE_FAULT",
            "ROVER_OFFLINE", "ROVER_ONLINE", "IMAGE_EVENT_DETECTED"} <= tipos
    assert nuvem.objects, "originais dos eventos subiram"
    assert json.dumps(rover.core.summary())
