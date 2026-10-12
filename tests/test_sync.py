"""Sync Engine: prioridade, chunks, retomada em 70 %, sem duplicar, backoff e recusas."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from valtexa_rover.config import RuntimeConfig, SyncConfig


def _grande(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    pixels[..., 0] = 255  # vermelho dominante → POSSÍVEL DOENÇA (original preservado)
    pixels[..., 1] //= 8
    pixels[..., 2] //= 8
    saida = io.BytesIO()
    Image.fromarray(pixels).save(saida, "JPEG", quality=95)
    return saida.getvalue()


def _rig_com_evento_grande(rig_factory, **kwargs):
    rig = rig_factory(["vermelho"], auto_sync=False,
                      config=RuntimeConfig(sync=SyncConfig(chunk_size=8192, backoff_base_s=1,
                                                           backoff_max_s=8)), **kwargs)
    arquivo = next(rig.images.iterdir())
    arquivo.write_bytes(_grande(1))
    rig.core.start_mission("m-1", "talhao-1")
    rig.run(1)
    return rig


def _item_original(rig):
    return rig.core.store.query(
        "SELECT q.* FROM sync_queue q JOIN files f ON f.id = q.ref_id "
        "WHERE q.kind = 'file' AND f.role = 'original'")[0]


def test_ordem_pela_prioridade_da_secao_19(rig_factory) -> None:
    rig = rig_factory(["verde", "verde", "vermelho", "verde"], auto_sync=False)
    rig.core.start_mission("m-1", "talhao-1")
    rig.run(4)
    ordem: list[int] = []
    original = rig.core.sync._request

    def espia(method, path, rel, **kwargs):
        if path == "/v1/device/edge/records":
            ordem.extend(r["body"].get("retention", {}).get("priority", 0)
                         for r in kwargs["json_body"]["records"] if r["kind"] == "result")
        return original(method, path, rel, **kwargs)

    rig.core.sync._request = espia
    rig.core.sync.run_until_idle()
    assert ordem == sorted(ordem) and ordem[0] == 2  # a doença sobe antes das normais
    assert rig.core.sync.pending() == 0


def test_queda_em_70_por_cento_retoma_de_70(rig_factory) -> None:
    rig = _rig_com_evento_grande(rig_factory)
    item = _item_original(rig)
    total = item["chunks_total"]
    assert total >= 10, "o arquivo precisa de vários chunks para o teste fazer sentido"
    setenta = int(total * 0.7)

    # Requisições antes dos chunks: sessão (versões) + registros + abertura do upload.
    puts: list[int] = []
    original = rig.cloud.__call__

    def conta(method, path, body, headers):
        if method == "PUT":
            puts.append(int(path.rsplit("/", 1)[1]))
            if len(puts) == setenta:
                rig.network.set_online(False)  # cai logo depois do chunk que fecha 70 %
        return original(method, path, body, headers)

    rig.network.handler = conta
    rig.core.sync.run_until_idle()
    item = _item_original(rig)
    assert item["status"] == "ENVIANDO" and item["chunks_done"] == setenta
    assert rig.core.sync.pending() > 0

    # Reinício do robô no meio, para provar que o progresso está no disco.
    rig.core = rig.new_core()
    rig.network.set_online(True)
    rig.core.sync.run_until_idle()
    item = _item_original(rig)
    assert item["status"] == "CONCLUIDO"
    assert sorted(puts) == list(range(total)), "cada chunk foi enviado exatamente uma vez"
    assert len(puts) == total
    sha = item["sha256"]
    assert sha in rig.cloud.objects and len(rig.cloud.objects) >= 1
    tipos = [r["type"] for r in rig.core.store.query("SELECT type FROM records")]
    assert "UPLOAD_STARTED" in tipos and "UPLOAD_COMPLETED" in tipos
    assert "ROVER_OFFLINE" in tipos


def test_reenvio_nao_duplica(rig_factory) -> None:
    rig = _rig_com_evento_grande(rig_factory)
    rig.core.sync.run_until_idle()
    registros, objetos, puts = len(rig.cloud.records), len(rig.cloud.objects), rig.cloud.chunk_puts
    # Reenvio forçado de tudo (ex.: o robô perdeu a confirmação da nuvem).
    with rig.core.store.transaction() as conn:
        conn.execute("UPDATE sync_queue SET status = 'PENDENTE', chunks_done = 0, "
                     "next_attempt_at = 0")
    rig.core.sync.run_until_idle()
    assert len(rig.cloud.records) == registros
    assert len(rig.cloud.objects) == objetos
    assert rig.cloud.chunk_puts == puts  # a nuvem já tinha o hash: nenhum chunk de novo
    assert rig.core.sync.pending() == 0


def test_backoff_exponencial_em_erro_do_servidor(rig_factory) -> None:
    rig = rig_factory(["verde"], auto_sync=False,
                      config=RuntimeConfig(sync=SyncConfig(backoff_base_s=2, backoff_max_s=60)))
    rig.core.start_mission("m-1", "talhao-1")
    rig.run(1)
    rig.core.sync.step(max_requests=1)  # só anuncia as versões
    rig.cloud.fail_queue = [503, 503, 503]
    esperas, tentativas = [], []
    for _ in range(3):
        rig.core.sync.step(max_requests=1)
        linha = rig.core.store.query(
            "SELECT attempts, next_attempt_at, status FROM sync_queue WHERE attempts > 0 "
            "ORDER BY attempts DESC LIMIT 1")[0]
        assert linha["status"] == "PENDENTE"
        tentativas.append(linha["attempts"])
        esperas.append(linha["next_attempt_at"] - rig.clock.monotonic())
        rig.clock.advance(esperas[-1])
    assert tentativas == [1, 2, 3]
    # base 2 s dobrando a cada tentativa, com jitter entre 50 % e 100 %
    for espera, teto in zip(esperas, (2, 4, 8), strict=True):
        assert teto / 2 <= espera <= teto
    rig.core.sync.run_until_idle()
    assert rig.core.sync.pending() == 0


def test_recusa_definitiva_vira_falhou_e_403_para_tudo(rig_factory) -> None:
    rig = rig_factory(["verde"], auto_sync=False)
    rig.core.start_mission("m-1", "talhao-1")
    rig.run(1)
    rig.core.sync.step(max_requests=1)  # só anuncia as versões
    rig.cloud.fail_queue = [422]
    rig.core.sync.step(max_requests=1)
    assert rig.core.sync.queue_status().get("FALHOU", 0) >= 1
    tipos = [r["type"] for r in rig.core.store.query("SELECT type FROM records")]
    assert "UPLOAD_FAILED" in tipos

    rig.cloud.fail_queue = [403]
    rel = rig.core.sync.step()
    assert rel.blocked and rig.core.sync.blocked
    enviados = rig.network.sent
    rig.core.sync.step()
    assert rig.network.sent == enviados  # não insiste em 403


def test_conflito_mesmo_id_conteudo_diferente(rig_factory) -> None:
    rig = rig_factory(["verde"], auto_sync=False)
    rig.core.start_mission("m-1", "talhao-1")
    rig.run(1)
    rid = rig.core.store.query("SELECT id FROM records WHERE kind = 'result'")[0]["id"]
    rig.cloud.records[rid] = "outro-hash"
    rig.core.sync.run_until_idle()
    linha = rig.core.store.query(
        "SELECT status, last_error FROM sync_queue WHERE ref_id = ?", (rid,))[0]
    assert linha["status"] == "FALHOU" and "conflito" in linha["last_error"]


def test_remaining_informa_quanto_falta(rig_factory) -> None:
    rig = rig_factory(["verde"] * 6, auto_sync=False,
                      config=RuntimeConfig(sync=SyncConfig(batch_size=2)))
    rig.core.start_mission("m-1", "talhao-1")
    rig.run(6)
    rig.core.sync.run_until_idle()
    assert rig.cloud.remaining_seen[0] > 0 and rig.cloud.remaining_seen[-1] == 0


def test_backoff_nao_prende_a_fila_depois_do_reinicio(rig_factory) -> None:
    """O *backoff* fica no relógio monotônico, que recomeça a cada boot.

    Antes da correção (Parte 5), um item com espera gravada em 90.000 s de uptime
    ficava preso por ~25 h depois de um reinício. O boot agora zera a espera.
    """
    rig = rig_factory(["verde"], auto_sync=False)
    rig.core.start_mission("m-1", "talhao-1")
    rig.run(1)
    with rig.core.store.transaction() as conn:
        conn.execute("UPDATE sync_queue SET next_attempt_at = 90000, attempts = 3")
    rig.core = rig.new_core()  # reinício: o boot roda o recover()
    assert rig.core.store.scalar(
        "SELECT MAX(next_attempt_at) FROM sync_queue WHERE status = 'PENDENTE'") == 0
    rig.core.sync.run_until_idle()
    assert rig.core.sync.pending() == 0
