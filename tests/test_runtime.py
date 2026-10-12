"""Missão offline-first, metadados da seção 16 e reinício no meio da missão."""

from __future__ import annotations

import json

import pytest

from valtexa_rover.hal.simulated import FolderCamera
from valtexa_rover.runtime.store import LocalStore

SEQ = ["verde", "verde", "verde", "vermelho", "verde", "verde", "azul", "verde", "cinza",
       "verde", "duvida", "verde"]


def _resultados(store: LocalStore) -> list[dict]:
    return [json.loads(r["payload"]) for r in
            store.query("SELECT payload FROM records WHERE kind = 'result' ORDER BY occurred_at")]


def test_missao_inteira_sem_rede(rig_factory) -> None:
    rig = rig_factory(SEQ)
    rig.core.models.refresh()
    rig.network.set_online(False)  # caiu antes de começar e não volta
    rig.core.start_mission("m-1", "talhao-1")
    rig.run(len(SEQ))
    rig.core.complete_mission()

    store = rig.core.store
    resultados = _resultados(store)
    assert len(resultados) == len(SEQ)
    tipos = [r["type"] for r in resultados]
    assert tipos.count("POSSIVEL_DOENCA") == 1 and tipos.count("POSSIVEL_PRAGA") == 1
    assert tipos.count("ANOMALIA") == 1
    eventos = store.query("SELECT type FROM records WHERE kind = 'event'")
    nomes = [e["type"] for e in eventos]
    assert nomes.count("IMAGE_EVENT_DETECTED") == 3
    for esperado in ("MISSION_STARTED", "MODEL_UPDATED", "MISSION_COMPLETED"):
        assert esperado in nomes
    assert rig.cloud.record_posts == 0 and rig.cloud.chunk_puts == 0  # nada saiu
    assert rig.core.sync.pending() > len(SEQ)
    assert store.scalar("SELECT COUNT(*) FROM telemetry") >= 2
    missao = rig.core.missions.get("m-1")
    assert missao.state == "CONCLUIDA" and missao.frames_done == len(SEQ) and missao.events == 3


def test_metadados_da_secao_16_em_cada_resultado(rig_factory) -> None:
    rig = rig_factory(["vermelho", "verde"])
    rig.core.start_mission("m-1", "talhao-1")
    rig.run(2)
    doenca, normal = _resultados(rig.core.store)
    for r in (doenca, normal):
        assert r["device_id"] == "dev-1" and r["mission_id"] == "m-1"
        assert r["field_id"] == "talhao-1" and r["occurred_at"]
        assert r["frame"]["camera_id"] == "cam-0"
        assert r["position"]["lat"] and r["position"]["fix"] == "fixed"
        assert r["model"] == {"key": "coffee-foliar", "version": "2.4.1", "sha256": rig.sha,
                              "artifact_kind": "otimizado", "dnaty_version": "2.2.1",
                              "target_hardware": "Rover Edge V1"}
        assert r["hardware"]["version"] == "Rover Edge V1 (simulado)"
        assert r["hardware"]["simulated"] is True
        assert "CPUExecutionProvider" in r["hardware"]["providers"]
        assert {"category", "label", "confidence", "top", "latency_ms"} <= set(r["result"])
        assert "temperature_c" in r["sensors"] and r["imu"] is not None
        assert r["retention"]["tier"] in {"ORIGINAL", "REDUZIDA", "METADADOS"}
    assert doenca["event_id"] and doenca["files"][0]["role"] == "original"
    assert doenca["retention"]["priority"] == 2
    assert normal["event_id"] is None and normal["files"][0]["role"] == "reduzida"


def test_buffer_circular_guarda_a_sequencia_do_evento(rig_factory) -> None:
    rig = rig_factory(["verde"] * 4 + ["vermelho"] + ["verde"] * 4)
    rig.core.start_mission("m-1", "talhao-1")
    rig.run(9)
    evento = rig.core.store.query(
        "SELECT id FROM records WHERE type = 'IMAGE_EVENT_DETECTED'")[0]["id"]
    contexto = rig.core.store.query(
        "SELECT role FROM files WHERE record_id = ? ORDER BY role", (evento,))
    assert sorted(r["role"] for r in contexto) == sorted(
        ["contexto-3", "contexto-2", "contexto-1", "contexto+1", "contexto+2", "contexto+3"])


def test_reinicio_no_meio_da_missao_continua_sem_perder_nem_duplicar(rig_factory) -> None:
    rig = rig_factory(["verde", "vermelho"] * 10)
    rig.network.set_online(False)
    rig.core.models.refresh()
    rig.core.start_mission("m-1", "talhao-1")
    rig.run(7)
    antes = rig.core.store.scalar("SELECT COUNT(*) FROM records WHERE kind = 'result'")
    # "Tira da tomada": nada de fechar com cuidado; abre outro Core sobre o mesmo disco.
    rig.core = rig.new_core()
    retomada = rig.core.missions.current()
    assert retomada.id == "m-1" and retomada.frames_done == 7
    assert rig.core.engine.loaded  # recarregou o modelo do registry, offline
    rig.run(5)
    assert rig.core.missions.current().frames_done == 12
    assert rig.core.store.scalar("SELECT COUNT(*) FROM records WHERE kind = 'result'") == \
        antes + 5
    tipos = [r["type"] for r in rig.core.store.query("SELECT type FROM records")]
    assert "MISSION_RESUMED" in tipos
    # Sem duplicata e tudo sobe quando a rede volta.
    rig.network.set_online(True)
    rig.core.complete_mission()
    rig.core.sync.run_until_idle()
    assert rig.core.sync.pending() == 0
    ids = [r["id"] for r in rig.core.store.query("SELECT id FROM records")]
    assert set(ids) <= set(rig.cloud.records) and len(set(ids)) == len(ids)


def test_queda_dentro_da_transacao_nao_deixa_quadro_pela_metade(rig_factory, monkeypatch) -> None:
    rig = rig_factory(["vermelho", "verde"])
    rig.core.start_mission("m-1", "talhao-1")
    rig.core.auto_sync = False

    def explode(*_a, **_k):
        raise RuntimeError("energia caiu")

    monkeypatch.setattr(rig.core.missions, "progress", explode)
    with pytest.raises(RuntimeError):
        rig.core.step()
    store = rig.core.store
    assert store.scalar("SELECT COUNT(*) FROM records WHERE kind = 'result'") == 0
    assert store.scalar("SELECT COUNT(*) FROM files") == 0
    assert store.scalar("SELECT COUNT(*) FROM records WHERE type = 'IMAGE_EVENT_DETECTED'") == 0
    assert store.disk_bytes() > 0  # o original foi ao disco antes do commit...
    rig.core = rig.new_core()  # ...e o boot o apaga (órfão)
    assert rig.core.recovered["orfaos"] == 1
    assert rig.core.store.disk_bytes() == 0


def test_camera_que_falha_vira_falha_ativa_e_a_missao_segue(rig_factory, tmp_path) -> None:
    rig = rig_factory(["verde"] * 4)
    rig.hardware.cameras = [FolderCamera("cam-0", rig.images, rig.clock, fail_on=[1, 2])]
    rig.core.start_mission("m-1", "talhao-1")
    relatorios = rig.run(4)
    assert sum(r.camera_failures for r in relatorios) == 2
    assert rig.core.missions.current().frames_done == 2
    falhas = rig.core.store.query("SELECT payload FROM records WHERE type = 'HARDWARE_FAULT'")
    assert len(falhas) == 1  # a mesma falha não gera um evento por quadro
    assert "camera:cam-0" not in rig.core.health.faults  # voltou


def test_telemetria_leva_saude_e_horimetro(rig_factory) -> None:
    rig = rig_factory(["verde"] * 12, telemetry_interval_s=5.0)
    rig.core.start_mission("m-1", "talhao-1")
    rig.run(12)
    rig.core.sync.run_until_idle()
    amostras = sorted(rig.cloud.telemetry.values(), key=lambda a: a["sequence_number"])
    assert len(amostras) >= 2
    ultima = amostras[-1]
    assert ultima["storage_total_bytes"] > 0 and ultima["storage_free_bytes"] > 0
    assert ultima["operating_hours"] > 0
    assert ultima["mission_id"] == "m-1" and ultima["lat"] is not None
    assert ultima["sensors"]["edge_ai_fps"] > 0
    assert [a["sequence_number"] for a in amostras] == sorted({a["sequence_number"]
                                                                for a in amostras})
