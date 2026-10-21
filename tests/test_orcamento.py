"""Orçamento de dados pela Starlink do Rover: contabilidade, corte, tempestade, deduplicação."""

from __future__ import annotations

import dataclasses
from datetime import timedelta

import pytest

from valtexa_rover.config import ConfigError, DataBudgetConfig, RuntimeConfig, SyncConfig
from valtexa_rover.hal.simulated import SimulatedController
from valtexa_rover.runtime.orcamento import (
    DeduplicadorPorPlanta,
    DetectorDeTempestade,
    classe_do_caminho,
)


def _config(**orcamento) -> RuntimeConfig:
    return RuntimeConfig(sync=SyncConfig(chunk_size=4096),
                         orcamento_dados=DataBudgetConfig(**orcamento))


def _fila(core, status: str | None = None) -> list:
    sql = ("SELECT q.*, f.role, r.kind AS rkind FROM sync_queue q "
           "LEFT JOIN files f ON f.id = q.ref_id LEFT JOIN records r ON r.id = q.ref_id")
    linhas = core.store.query(sql)
    return [linha for linha in linhas if status is None or linha["status"] == status]


def _eventos(core, tipo: str) -> list:
    return core.store.query("SELECT payload FROM records WHERE kind = 'event' AND type = ?",
                            (tipo,))


class _CameraDeLado:
    """A câmera da pasta, dizendo de que lado do Rover está (a deduplicação usa a rua dele)."""

    def __init__(self, camera, lado: str) -> None:
        self._camera, self.lado = camera, lado
        self.camera_id = camera.camera_id

    def capture(self):
        return dataclasses.replace(self._camera.capture(), side=self.lado)

    def __getattr__(self, nome):
        return getattr(self._camera, nome)


# -- peças isoladas -----------------------------------------------------------------------


def test_classe_pelo_caminho_do_protocolo() -> None:
    assert classe_do_caminho("/v1/device/edge/records") == "registros"
    assert classe_do_caminho("/v1/device/uploads/up-1/chunks/3") == "arquivos"
    assert classe_do_caminho("/v1/device/sync") == "telemetria"
    assert classe_do_caminho("/v1/device/commands?limit=5") == "comando"
    assert classe_do_caminho("/v1/device/teleop") == "comando"
    assert classe_do_caminho("/v1/device/missions/m-1/ack") == "missao"
    assert classe_do_caminho("/v1/device/ota/pacote") == "atualizacao"
    assert classe_do_caminho("/v1/device/priores") == "atualizacao"
    assert classe_do_caminho("/v1/qualquer") == "outros"


def test_tempestade_liga_com_histerese_e_limita_originais() -> None:
    d = DetectorDeTempestade(janela_s=60, min_quadros=10, fracao=0.5, fracao_saida=0.2,
                             originais_por_janela=2)
    t = 0.0
    for _ in range(9):  # poucos quadros: não julga
        assert d.observa(True, t) is None
        t += 1
    assert d.observa(True, t) == "inicio"
    assert [d.permite_original(t) for _ in range(4)] == [True, True, False, False]
    # 0,3 de eventos fica acima da saída (0,2): não pisca
    for i in range(60):
        t += 1
        mudou = d.observa(i % 10 < 3, t)
        assert mudou is None
    for _ in range(60):
        t += 1
        mudou = d.observa(False, t)
        if mudou:
            break
    assert mudou == "fim" and not d.ativa and d.permite_original(t)


def test_deduplicacao_guarda_os_k_melhores_da_planta() -> None:
    d = DeduplicadorPorPlanta(k=2, espacamento_m=0.7)
    assert d.chave({"x_m": 3.0}, "left") is None  # sem a rua não sabe a planta
    assert d.chave({"row_left": "L3", "x_m": 3.0}, None) is None  # câmera sem lado
    chave = d.chave({"row_left": "L3", "row_right": "L4", "x_m": 3.0}, "left")
    assert chave == ("L3", 4)
    assert d.chave({"row_left": "L3", "x_m": 3.1}, "left") == chave  # a mesma planta
    assert d.decide(chave, 0.70, ("a",)) == (True, [])
    assert d.decide(chave, 0.80, ("b",)) == (True, [])
    assert d.decide(chave, 0.60, ("c",)) == (False, [])  # pior que os dois
    sobe, saiu = d.decide(chave, 0.95, ("d",))
    assert sobe and [e.record_ids for e in saiu] == [("a",)]  # o pior sai do pódio
    assert DeduplicadorPorPlanta(0, 0.7).chave({"row_left": "L3", "x_m": 1}, "left") is None


def test_config_valida_e_mescla_as_fracoes() -> None:
    cfg = RuntimeConfig.from_dict(
        {"orcamento_dados": {"bytes_dia": 10, "fracao_por_prioridade": {"SECUNDARIO": 0.1}}})
    assert cfg.orcamento_dados.fracao_por_prioridade["NORMAL_COMPACTADA"] == 0.4
    assert cfg.orcamento_dados.fracao_por_prioridade["SECUNDARIO"] == 0.1
    with pytest.raises(ConfigError, match="não pode ter fração maior"):
        RuntimeConfig.from_dict({"orcamento_dados": {"fracao_por_prioridade": {
            "NORMAL_COMPACTADA": 0.55}}})
    with pytest.raises(ConfigError, match="desconhecida"):
        RuntimeConfig.from_dict({"orcamento_dados": {"fracao_por_prioridade": {"X": 0.1}}})
    with pytest.raises(ConfigError, match="desconhecida"):
        RuntimeConfig.from_dict({"orcamento_dados": {"teto": 1}})
    assert RuntimeConfig().to_dict()["orcamento_dados"]["bytes_dia"] == 2_000_000_000


# -- no Rover -----------------------------------------------------------------------------


def test_contabilidade_conta_o_que_trafegou_por_classe(rig_factory) -> None:
    rig = rig_factory(["verde", "vermelho", "verde"], auto_sync=False)
    trafego = {"enviados": 0, "requisicoes": 0}
    original = rig.cloud.__call__

    def mede(method, path, body, headers):
        trafego["enviados"] += len(body) + sum(len(k) + len(v) + 4 for k, v in headers.items())
        trafego["requisicoes"] += 1
        return original(method, path, body, headers)

    rig.network.handler = mede
    antes = rig.core.orcamento.relatorio()["enviados_bytes"]  # o boot já falou com a nuvem
    rig.core.start_mission("m-1", "talhao-1")
    rig.run(3)
    rig.core.sync.run_until_idle()
    rel = rig.core.orcamento.relatorio()
    assert {"registros", "arquivos", "telemetria", "missao"} <= set(rel["por_classe"])
    sobrecarga = rig.config.orcamento_dados.sobrecarga_requisicao_bytes
    assert rel["enviados_bytes"] - antes == (trafego["enviados"]
                                             + sobrecarga * trafego["requisicoes"])
    assert rel["recebidos_bytes"] > 0 and rel["usado_bytes"] == (
        rel["enviados_bytes"] + rel["recebidos_bytes"])
    assert rel["origem_do_teto"] == "rover.json" and rel["fracao_usada"] < 0.01
    # a missão leva o que é dela (os arquivos e registros foram atribuídos a m-1)
    assert rig.core.orcamento.usado_missao("m-1") > rel["por_classe"]["arquivos"]["enviados"]
    saude = rig.core.health.relatorio(online=True)["dados"]
    assert saude["usado_bytes"] == rel["usado_bytes"] and saude["tempestade"] is False


def test_teto_corta_normal_primeiro_e_nunca_corta_evento(rig_factory) -> None:
    # teto minúsculo: a primeira rodada da fila já passa de 60% (NORMAL espera)
    rig = rig_factory(["verde", "verde", "vermelho", "verde"], auto_sync=False,
                      config=_config(bytes_dia=40_000))
    rig.core.start_mission("m-1", "talhao-1")
    rig.run(4)
    rig.core.sync.run_until_idle()
    pendentes = _fila(rig.core, "PENDENTE")
    assert pendentes, "com o teto baixo alguma coisa precisa esperar"
    # o que espera é cortável: nada de evento, nada de prioridade 1
    assert all(q["rkind"] != "event" and q["priority"] > 1 for q in pendentes)
    # o original da doença (prioridade 2) subiu antes das reduzidas NORMAL (5)
    originais = [q for q in _fila(rig.core) if q["role"] == "original"]
    assert originais and all(q["status"] == "CONCLUIDO" for q in originais)
    assert any(q["role"] == "reduzida" for q in pendentes)
    assert rig.core.sync.saude()["deferred"] == len(pendentes)
    assert rig.core.orcamento.relatorio()["prioridade_maxima_liberada"] < 5
    # dia novo na fazenda: teto novo, a fila esvazia
    rig.clock.advance(24 * 3600)
    rig.core.sync.run_until_idle()
    assert not _fila(rig.core, "PENDENTE")


def test_teto_estourado_ainda_manda_telemetria_eventos_e_alerta(rig_factory) -> None:
    rig = rig_factory(["vermelho", "verde"], auto_sync=False, config=_config(bytes_dia=1))
    rig.core.start_mission("m-1", "talhao-1")
    rig.run(2)
    rig.core.sync.run_until_idle()
    assert rig.core.store.scalar("SELECT COUNT(*) FROM telemetry WHERE sent = 0") == 0
    assert all(q["status"] == "CONCLUIDO" for q in _fila(rig.core) if q["rkind"] == "event")
    resultados = [q for q in _fila(rig.core) if q["rkind"] == "result"]
    assert resultados and all(q["status"] == "PENDENTE" for q in resultados)
    alertas = _eventos(rig.core, "DATA_BUDGET")
    assert alertas, "estourar o teto avisa a plataforma"
    assert any(q["status"] == "CONCLUIDO" for q in _fila(rig.core)
               if q["rkind"] == "event" and q["priority"] == 1)


def test_teto_da_plataforma_vale_por_cima_do_rover_json(rig_factory) -> None:
    rig = rig_factory(["verde"], auto_sync=False, config=_config(bytes_dia=10_000_000))
    orc = rig.core.orcamento
    assert not orc.aplicar_da_plataforma({"bytes_dia": -1})
    assert not orc.aplicar_da_plataforma({"bytes_dia": True})
    assert not orc.aplicar_da_plataforma("lixo")
    rig.core.start_mission("m-1", "talhao-1",
                           {"command": {"orcamento_dados": {"bytes_dia": 123_456,
                                                            "bytes_missao": 50_000}}})
    assert orc.config.bytes_dia == 123_456 and orc.config.bytes_missao == 50_000
    assert orc.relatorio()["origem_do_teto"] == "plataforma"
    # persiste: o Core novo (reinício) lê o mesmo teto
    assert rig.new_core().orcamento.config.bytes_dia == 123_456


def test_teto_da_missao_segura_a_missao(rig_factory) -> None:
    rig = rig_factory(["verde", "verde", "verde"], auto_sync=False,
                      config=_config(bytes_dia=0, bytes_missao=20_000))
    rig.core.start_mission("m-1", "talhao-1")
    rig.run(3)
    rig.core.sync.run_until_idle()
    assert rig.core.orcamento.usado_missao("m-1") >= 0.6 * 20_000
    assert any(q["role"] == "reduzida" for q in _fila(rig.core, "PENDENTE"))


def test_tempestade_retem_originais_e_contexto_e_avisa(rig_factory) -> None:
    rig = rig_factory(["vermelho"] * 30, auto_sync=False,
                      config=_config(tempestade_min_quadros=10, tempestade_janela_s=600,
                                     tempestade_originais_por_janela=2, k_por_planta=0))
    rig.core.start_mission("m-1", "talhao-1")
    rig.run(30)
    rig.core.complete_mission()
    storm = _eventos(rig.core, "EVENT_STORM")
    assert len(storm) == 1 and '"estado":"inicio"' in storm[0]["payload"]
    originais = [q for q in _fila(rig.core) if q["role"] == "original"]
    retidos = [q for q in originais if q["status"] == "RETIDO"]
    # 9 antes de ligar; o que liga e o seguinte sobem (2 por janela); o resto fica no disco
    assert len(originais) == 30 and len(retidos) == 30 - 9 - 2
    assert {q["last_error"] for q in retidos} == {"retido: tempestade"}
    contexto = [q for q in _fila(rig.core) if (q["role"] or "").startswith("contexto")]
    assert contexto
    assert any(q["last_error"] == "retido: tempestade_contexto" for q in contexto)
    # o resultado (metadados e recortes para a fusão) sobe de todos os 30
    assert sum(1 for q in _fila(rig.core) if q["rkind"] == "result") == 30
    assert all(q["status"] != "RETIDO" for q in _fila(rig.core) if q["kind"] == "record")
    # retido não é apagado: o arquivo continua no disco
    caminho = rig.core.store.query("SELECT path FROM files WHERE id = ?",
                                   (retidos[0]["ref_id"],))[0]["path"]
    assert rig.core.store.blob_path(caminho).exists()
    rel = rig.core.orcamento.relatorio()
    # o contexto dos eventos retidos fica com eles; o dos que subiram, só na tempestade
    assert rel["retidos"]["tempestade"]["itens"] == sum(
        1 for q in _fila(rig.core) if q["last_error"] == "retido: tempestade")
    assert rel["retidos"]["tempestade"]["itens"] > len(retidos)


def test_deduplicacao_por_planta_no_rover(rig_factory) -> None:
    rig = rig_factory(["vermelho"] * 6, auto_sync=False,
                      config=_config(k_por_planta=2, tempestade_min_quadros=1000))
    rig.hardware.controller = SimulatedController({"row_left": "L7", "x_m": 12.0})
    rig.hardware.cameras[0] = _CameraDeLado(rig.hardware.cameras[0], "left")
    rig.core.start_mission("m-1", "talhao-1")
    rig.run(6)
    rig.core.complete_mission()
    originais = [q for q in _fila(rig.core) if q["role"] == "original"]
    assert len(originais) == 6
    assert sum(q["status"] == "PENDENTE" for q in originais) == 2
    assert {q["last_error"] for q in originais if q["status"] == "RETIDO"} == {
        "retido: dedup_planta"}
    # o contexto dos retidos também fica (só os k sobem com a sequência)
    contexto_pendente = {q["ref_id"] for q in _fila(rig.core, "PENDENTE")
                         if (q["role"] or "").startswith("contexto")}
    eventos_que_sobem = {r["record_id"] for r in rig.core.store.query(
        "SELECT f.record_id FROM files f JOIN sync_queue q ON q.ref_id = f.id "
        "WHERE q.status = 'PENDENTE' AND f.role LIKE 'contexto%'")}
    assert len(eventos_que_sobem) <= 2 and contexto_pendente


def test_rebaixado_pela_deduplicacao_sai_da_fila_se_nao_comecou(rig_factory) -> None:
    rig = rig_factory(["verde"], auto_sync=False, config=_config(k_por_planta=1))
    orc = rig.core.orcamento
    rig.core.start_mission("m-1", "talhao-1")
    rig.run(1)
    reduzida = _fila(rig.core, "PENDENTE")
    file_q = next(q for q in reduzida if q["kind"] == "file")
    record_id = rig.core.store.scalar("SELECT record_id FROM files WHERE id = ?",
                                      (file_q["ref_id"],))
    with rig.core.store.transaction() as conn:
        assert orc.reter(conn, (record_id,), "dedup_planta") == 1
    assert rig.core.store.scalar("SELECT status FROM sync_queue WHERE id = ?",
                                 (file_q["id"],)) == "RETIDO"
    # a plataforma pede de volta
    assert orc.liberar_retidos(motivo="dedup_planta") == 1
    rig.core.sync.run_until_idle()
    assert rig.core.store.scalar("SELECT status FROM sync_queue WHERE id = ?",
                                 (file_q["id"],)) == "CONCLUIDO"


def test_adiado_demais_vira_amostra(rig_factory) -> None:
    rig = rig_factory(["verde"] * 10, auto_sync=False,
                      config=_config(bytes_dia=1, amostra_adiados_cada=4, adiado_max_h=1))
    rig.core.start_mission("m-1", "talhao-1")
    rig.run(10)
    rig.core.sync.run_until_idle()
    assert len([q for q in _fila(rig.core, "PENDENTE") if q["kind"] == "file"]) == 10
    rig.clock.advance(2 * 3600)
    saida = rig.core.purge()
    assert saida["amostra_orcamento"] == 10 - 3  # ficam o 1º, o 5º e o 9º
    sobra = [q for q in _fila(rig.core, "PENDENTE") if q["kind"] == "file"]
    assert len(sobra) == 3
    assert rig.core.orcamento.relatorio()["retidos"]["amostra_orcamento"]["itens"] == 7


def test_relatorio_sobe_no_fim_da_missao(rig_factory) -> None:
    rig = rig_factory(["verde", "verde"], auto_sync=True)
    rig.core.start_mission("m-1", "talhao-1")
    rig.run(2)
    rig.core.complete_mission()
    fim = _eventos(rig.core, "MISSION_COMPLETED")[0]["payload"]
    assert '"dados":' in fim and '"usado_missao_bytes":' in fim
    assert rig.core.orcamento.missao_ativa is None


def test_dia_do_orcamento_e_o_da_fazenda(rig_factory, clock) -> None:
    rig = rig_factory(["verde"], auto_sync=False)
    orc = rig.core.orcamento
    agora = clock.now()
    inicio_dia_utc = agora.replace(hour=1, minute=0, second=0, microsecond=0)
    # 01h UTC é 22h do dia anterior em Brasília
    assert orc.dia(inicio_dia_utc) == (inicio_dia_utc - timedelta(days=1)).date().isoformat()


def test_comando_da_plataforma_libera_e_relata(rig_factory) -> None:
    from valtexa_rover.runtime.comandos import CommandExecutor

    rig = rig_factory(["vermelho"] * 4, auto_sync=False,
                      config=_config(tempestade_min_quadros=2, tempestade_fracao=0.5,
                                     tempestade_originais_por_janela=0, k_por_planta=0))
    rig.core.start_mission("m-1", "talhao-1")
    rig.run(4)
    executor = CommandExecutor(rig.core)
    ok, resultado, erro = executor.execute("orcamento", {})
    assert ok and resultado["retidos"]["tempestade"]["itens"] >= 2
    ok, _r, erro = executor.execute("liberar_retidos", {"motivo": "inventado"})
    assert not ok and "motivo desconhecido" in erro
    ok, resultado, _e = executor.execute("liberar_retidos", {"motivo": "tempestade",
                                                             "maximo": 1})
    assert ok and resultado["liberados"] == 1


def test_registros_em_gzip_gastam_menos_starlink(rig_factory, tmp_path) -> None:
    usados = {}
    for comprimir in (False, True):
        rig = rig_factory(["verde"] * 20, auto_sync=False, data_dir=tmp_path / f"d{comprimir}",
                          config=RuntimeConfig(sync=SyncConfig(chunk_size=4096,
                                                               comprimir_registros=comprimir)))
        rig.core.start_mission("m-1", "talhao-1")
        rig.run(20)
        rig.core.sync.run_until_idle()
        assert len(rig.cloud.records) >= 20  # a nuvem leu (e conferiu a assinatura) igual
        rel = rig.core.orcamento.relatorio()
        usados[comprimir] = rel["por_classe"]["registros"]["enviados"]
    assert usados[True] < 0.8 * usados[False]


def test_estimativa_simulada_roda_e_o_teto_segura(tmp_path) -> None:
    """A simulação do dia (docs): com o teto, sobe perto dele; sem, a banda é o limite."""
    from valtexa_rover.sim.orcamento import Cenario, Hipoteses, simular

    h = Hipoteses(horas=0.5)
    cfg = DataBudgetConfig(bytes_dia=200_000_000)
    antes = simular(Cenario("t", 1.0, tempestade=True), h, None)
    depois = simular(Cenario("t", 1.0, tempestade=True), h, cfg)
    assert antes.total_subiu > 3 * depois.total_subiu
    assert depois.total_subiu <= 1.2 * cfg.bytes_dia
    assert depois.retido > 0 and depois.retido_itens.get("tempestade", 0) > 0


# -- verificação hostil da Parte 1 (rodada 4, parte 2) ------------------------------------


def test_requisicao_que_cai_depois_de_abrir_conta_no_orcamento(rig_factory) -> None:
    """Um chunk que esgota o tempo à espera da resposta já saiu pela Starlink: conta."""
    from valtexa_rover.hal.interfaces import NetworkUnavailableError

    rig = rig_factory(["verde"], auto_sync=False)
    orc = rig.core.orcamento
    antes = orc.usado_dia()

    def cai_no_meio(method, path, body, headers):
        raise NetworkUnavailableError("tempo esgotado", pode_ter_enviado=True)

    rig.network.handler = cai_no_meio
    with pytest.raises(NetworkUnavailableError):
        rig.core.client.request("POST", "/v1/device/uploads/x/chunks/0", raw=b"z" * 50_000)
    depois = orc.usado_dia()
    assert depois - antes >= 50_000
    assert orc.relatorio()["por_classe"]["arquivos"]["recebidos"] == 0

    # sem conexão (DNS, recusa): o corpo não saiu, não conta
    def recusa(method, path, body, headers):
        raise NetworkUnavailableError("recusada")

    rig.network.handler = recusa
    with pytest.raises(NetworkUnavailableError):
        rig.core.client.request("POST", "/v1/device/uploads/x/chunks/0", raw=b"z" * 50_000)
    assert orc.usado_dia() == depois


def test_http_distingue_falha_antes_e_depois_de_abrir() -> None:
    import socket
    import urllib.error

    from valtexa_rover.hal.local import _pode_ter_enviado

    assert not _pode_ter_enviado(urllib.error.URLError(socket.gaierror("dns")))
    assert not _pode_ter_enviado(urllib.error.URLError(ConnectionRefusedError()))
    assert _pode_ter_enviado(urllib.error.URLError(TimeoutError()))
    assert _pode_ter_enviado(TimeoutError())
    assert _pode_ter_enviado(ConnectionResetError())


def test_cliente_extra_do_video_conta_no_orcamento(rig_factory) -> None:
    """O vídeo ao vivo tem cliente próprio (thread); antes ele subia sem contar."""
    rig = rig_factory(["verde"], auto_sync=False)
    antes = rig.core.orcamento.relatorio()["por_classe"].get("video", {"enviados": 0})
    cliente = rig.core.novo_cliente()
    assert cliente is not rig.core.client and cliente.contador is not None
    cliente.request("POST", "/v1/device/video/quadro", raw=b"j" * 30_000,
                    content_type="image/jpeg")
    video = rig.core.orcamento.relatorio()["por_classe"]["video"]
    assert video["enviados"] - antes["enviados"] >= 30_000


def test_cli_usa_o_cliente_contado_para_o_video() -> None:
    import inspect

    from valtexa_rover import cli

    fonte = inspect.getsource(cli)
    assert "core.novo_cliente()" in fonte
    assert "DeviceClient(core.identity" not in fonte


def test_prioridade_1_nunca_e_retida_nem_pela_tempestade(rig_factory) -> None:
    from valtexa_rover.config import Category, RetentionConfig, RetentionRule, Tier

    regras = RetentionConfig().rules
    regras[Category.POSSIVEL_DOENCA] = RetentionRule(Tier.ORIGINAL, 90, 1,
                                                     preserve_sequence=True)
    cfg = dataclasses.replace(
        _config(tempestade_min_quadros=3, tempestade_janela_s=600,
                tempestade_originais_por_janela=0, k_por_planta=1),
        retencao=RetentionConfig(rules=regras))
    rig = rig_factory(["vermelho"] * 8, auto_sync=False, config=cfg)
    rig.core.start_mission("m-1", "talhao-1")
    rig.run(8)
    assert rig.core.orcamento.tempestade.ativa
    assert not [q for q in _fila(rig.core, "RETIDO") if q["priority"] <= 1]
    # e o reter direto também não pega prioridade 1
    rid = rig.core.store.scalar(
        "SELECT f.record_id FROM files f JOIN sync_queue q ON q.ref_id = f.id "
        "WHERE q.priority = 1 LIMIT 1")
    if rid:
        with rig.core.store.transaction() as conn:
            assert rig.core.orcamento.reter(conn, (rid,), "dedup_planta") == 0
