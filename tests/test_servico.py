"""``valtexa-rover servico``: o laço que o systemd sobe (boot → missão → captura → fim)."""

from __future__ import annotations

import json
from pathlib import Path

from conftest import FakeCloud, designation

from valtexa_rover import cli
from valtexa_rover.clock import SystemClock
from valtexa_rover.hal.rover_v3 import NavegacaoPorArquivo
from valtexa_rover.hal.simulated import FlakyNetwork
from valtexa_rover.runtime.client import DeviceIdentity, generate_keypair
from valtexa_rover.runtime.core import RoverCore
from valtexa_rover.runtime.model_manager import FolderArtifactSource
from valtexa_rover.runtime.store import LocalStore
from valtexa_rover.sim import MODELO_DEMO_SHA256, hal_de_demonstracao

NUVEM: FakeCloud | None = None


def fabrica_de_teste(*, dados: Path, url: str, clock):
    """A HAL de demonstração com a rede trocada pelo dublê (sem servidor HTTP no teste)."""
    hardware = hal_de_demonstracao(dados=dados, url=url, clock=clock)
    hardware.network = FlakyNetwork(NUVEM)
    return hardware


def test_servico_roda_sem_missao_e_conclui_a_local(tmp_path, monkeypatch) -> None:
    global NUVEM
    privada, publica = generate_keypair()
    NUVEM = FakeCloud(publica, [designation("talhao-1", MODELO_DEMO_SHA256)])
    ident = tmp_path / "identity.json"
    DeviceIdentity("dev-1", "dk", privada).save(ident)
    config = tmp_path / "rover.json"
    config.write_text(json.dumps({"sync": {"chunk_size": 4096}}))
    # Sem chave da plataforma o robô não aceita missão da nuvem (só registra o motivo);
    # o laço segue: telemetria e sincronização continuam.
    codigo = cli.main(["servico", "--dados", str(tmp_path / "dados"), "--identidade", str(ident),
                       "--config", str(config), "--hal", "test_servico:fabrica_de_teste",
                       "--url", "http://dublê", "--intervalo", "0", "--max-ciclos", "5"])
    assert codigo == 0
    assert NUVEM.telemetry, "o serviço subiu telemetria"
    assert (tmp_path / "dados" / "ota" / "models" / f"{MODELO_DEMO_SHA256}.onnx").exists()


def test_controlador_diz_quando_a_rota_acabou(clock, tmp_path) -> None:
    hardware = hal_de_demonstracao(dados=tmp_path, url="http://x", clock=clock)
    assert hardware.controller.state()["route_done"] is False
    clock.advance(hardware.controller.comprimento_m / 0.7 + 1)
    assert hardware.controller.state()["route_done"] is True
    assert hardware.simulated


def fabrica_com_navegacao(*, dados: Path, url: str, clock):
    """A HAL de demonstração com o controlador do Rover V3 (arquivos em ``dados/run``)."""
    hardware = fabrica_de_teste(dados=dados, url=url, clock=clock)
    run = Path(dados) / "run"
    hardware.controller = NavegacaoPorArquivo(run / "navegacao.json", run / "parar.json",
                                              clock=clock)
    return hardware


def test_navegacao_por_arquivo_encerra_so_a_propria_missao(tmp_path) -> None:
    global NUVEM
    privada, publica = generate_keypair()
    NUVEM = FakeCloud(publica, [designation("talhao-1", MODELO_DEMO_SHA256)])
    ident = tmp_path / "identity.json"
    DeviceIdentity("dev-1", "dk", privada).save(ident)
    dados = tmp_path / "dados"
    clock = SystemClock()
    core = RoverCore(fabrica_com_navegacao(dados=dados, url="http://dublê", clock=clock), dados,
                     DeviceIdentity.load(ident), clock=clock,
                     artifact_source=FolderArtifactSource(dados / "ota" / "models"))
    core.boot()
    core.start_mission("m-1", "talhao-1", {"route": [[-47.06, -22.90], [-47.05, -22.90]]})
    core.close()
    run = dados / "run"
    run.mkdir()
    (run / "navegacao.json").write_text(
        '{"mission_id": "m-0", "route_done": true, "row_left": "L03", "x_m": 12.5, "v": 1}')

    def servico() -> str:
        assert cli.main(["servico", "--dados", str(dados), "--identidade", str(ident),
                         "--hal", "test_servico:fabrica_com_navegacao", "--url", "http://dublê",
                         "--intervalo", "0", "--max-ciclos", "3"]) == 0
        banco = LocalStore(dados)
        try:
            return banco.query("SELECT state FROM missions WHERE id = 'm-1'")[0]["state"]
        finally:
            banco.close()

    assert servico() == "EM_EXECUCAO"  # route_done de outra missão: ignorado
    banco = LocalStore(dados)
    try:
        corpos = [json.loads(r["payload"]) for r in banco.query(
            "SELECT payload FROM records WHERE kind = 'result'")]
    finally:
        banco.close()
    assert corpos and all(c["navigation"] == {"row_left": "L03", "x_m": 12.5} for c in corpos)
    anunciada = json.loads((run / "missao.json").read_text(encoding="utf-8"))
    assert anunciada["id"] == "m-1" and anunciada["payload"]["route"][0] == [-47.06, -22.90]

    (run / "navegacao.json").write_text('{"mission_id": "m-1", "route_done": true}')
    assert servico() != "EM_EXECUCAO"
    assert json.loads((run / "missao.json").read_text(encoding="utf-8"))["id"] is None
