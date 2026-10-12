"""O comando assinado ``ensaio``: thread, progresso, cancelamento, laudo e recusas."""

from __future__ import annotations

import json
import threading
import time
from datetime import timedelta

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from test_comandos import CentroDeComando

from valtexa_rover.clock import SystemClock
from valtexa_rover.ensaios.simulado import bancada
from valtexa_rover.hal.interfaces import HttpResponse
from valtexa_rover.runtime.client import DeviceIdentity
from valtexa_rover.runtime.comandos import CommandExecutor
from valtexa_rover.runtime.mission import canonical_command


class CentroComEnsaio(CentroDeComando):
    """O centro de comando dos testes + progresso e o eco de rede da plataforma."""

    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)
        self.progresso: dict[str, list[dict]] = {}

    def __call__(self, method, path, body, headers):
        if path.startswith("/v1/device/commands/") and path.endswith("/progresso"):
            self.progresso.setdefault(path.split("/")[4], []).append(json.loads(body))
            return HttpResponse(200, b"{}")
        if path.startswith("/v1/device/ensaio/eco"):
            if method == "POST":
                return HttpResponse(200, json.dumps({"recebidos": len(body)}).encode())
            n = int(path.partition("bytes=")[2] or 0)
            return HttpResponse(200, b"\0" * n)
        return super().__call__(method, path, body, headers)


def monta(rig, tmp_path):
    centro = CentroComEnsaio(rig.identity.device_id, rig.cloud)
    rig.identity = DeviceIdentity(rig.identity.device_id, rig.identity.key_id,
                                  rig.identity.private_key, centro.publica)
    rig.network.handler = centro
    hardware, _ = bancada(tmp_path / "bancada", rig.clock, cameras=2)
    hardware.network = rig.network  # a rede do robô fala com o centro
    rig.hardware = hardware
    rig.core = rig.new_core()
    return centro, CommandExecutor(rig.core, tmp_path / "rover.json")


@pytest.fixture
def robo(rig_factory, tmp_path):
    rig = rig_factory(["verde"])
    centro, executor = monta(rig, tmp_path)
    return rig, centro, executor


def espera_resultado(centro, cid, executor, limite_s=10.0) -> dict:
    executor.ensaios.aguardar(limite_s)
    assert cid in centro.resultados, "o laudo não chegou"
    return centro.resultados[cid]


def test_ensaio_assinado_roda_na_thread_e_devolve_o_laudo(robo) -> None:
    rig, centro, executor = robo
    imu = rig.hardware.imu
    leitura_original, libera = imu.read, threading.Event()

    def segura():  # a IMU só responde depois que o segundo comando chegar
        libera.wait(10)
        return leitura_original()

    imu.read = segura
    cid = centro.manda("ensaio", {"ensaio": "imu", "params": {"duracao_s": 2}}, rig.clock)
    rede = centro.manda("ensaio", {"ensaio": "rede", "params": {"pings": 3}}, rig.clock)
    assert executor.poll() == 2
    libera.set()
    resposta = espera_resultado(centro, cid, executor)
    assert resposta["ok"] and resposta["result"]["status"] == "aprovado"
    assert resposta["result"]["simulado"] is True
    assert "iniciado" in [p["etapa"] for p in centro.progresso[cid]]
    # O segundo chegou com o primeiro em curso: recusado, com o motivo.
    assert resposta is not centro.resultados.get(rede)
    assert not centro.resultados[rede]["ok"] and "em curso" in centro.resultados[rede]["error"]


def test_ensaio_de_rede_usa_o_canal_assinado(robo) -> None:
    rig, centro, executor = robo
    cid = centro.manda("ensaio", {"ensaio": "rede", "params": {"pings": 3, "carga_kb": 4,
                                                               "vazao_min_kbps": 0}}, rig.clock)
    executor.poll()
    laudo = espera_resultado(centro, cid, executor)["result"]
    assert laudo["medidas"]["perda_pct"] == 0
    assert laudo["amostras"]["subida_bytes"] == laudo["amostras"]["descida_bytes"] == 4096


def test_ensaio_repetido_ou_forjado_e_recusado(robo) -> None:
    rig, centro, executor = robo
    centro.manda("ensaio", {"ensaio": "imu", "params": {"duracao_s": 1}}, rig.clock)
    copia = dict(centro.fila[-1])
    assert executor.poll() == 1
    executor.ensaios.aguardar(10)
    centro.fila.append(copia)  # o mesmo envelope de novo
    forjado = centro.envelope("device.ensaio", {"command_id": "x", "kind": "ensaio",
                                                "params": {"ensaio": "parada"}}, 50, rig.clock)
    forjado["signature"] = Ed25519PrivateKey.generate().sign(
        canonical_command(forjado["payload"])).hex()
    centro.fila.append(forjado)
    assert executor.poll() == 0
    assert executor.ensaios.ocupado is None and "x" not in centro.resultados


def test_parametro_invalido_e_missao_em_curso_recusam(robo) -> None:
    rig, centro, executor = robo
    ruim = centro.manda("ensaio", {"ensaio": "gnss", "params": {"taxa_hz": 0}}, rig.clock)
    estranho = centro.manda("ensaio", {"ensaio": "voar"}, rig.clock)
    executor.poll()
    assert "taxa_hz" in centro.resultados[ruim]["error"]
    assert "voar" in centro.resultados[estranho]["error"]
    rig.core.start_mission("m-1", "talhao-1")
    em_missao = centro.manda("ensaio", {"ensaio": "imu"}, rig.clock)
    executor.poll()
    assert "missão em curso" in centro.resultados[em_missao]["error"]


def test_laudo_sem_rede_fica_guardado_e_vai_depois(robo) -> None:
    rig, centro, executor = robo
    rig.network.set_online(False)
    executor._fim_ensaio("c-9", {"ensaio": "imu", "status": "aprovado", "erro": None})
    assert executor._laudos_pendentes and "c-9" not in centro.resultados
    rig.network.set_online(True)
    executor.poll()
    assert centro.resultados["c-9"]["result"]["status"] == "aprovado"
    assert not executor._laudos_pendentes


# -- com o relógio do sistema: o laço do Runtime e o ensaio correm juntos ---------------------


@pytest.fixture
def robo_real(tmp_path, model_repo):
    from conftest import Rig

    rig = Rig(tmp_path, SystemClock(), model_repo, ["verde"])
    centro, executor = monta(rig, tmp_path)
    return rig, centro, executor


def test_cancelamento_assinado_para_o_ensaio(robo_real) -> None:
    rig, centro, executor = robo_real
    cid = centro.manda("ensaio", {"ensaio": "homem_morto", "params": {"timeout_s": 60}},
                       rig.clock)
    executor.poll()
    assert executor.ensaios.ocupado == cid
    cancela = centro.manda("ensaio_cancelar", {"command_id": cid}, rig.clock)
    executor.poll()
    assert centro.resultados[cancela]["result"] == {"command_id": cid, "cancelado": True}
    laudo = espera_resultado(centro, cid, executor)["result"]
    assert laudo["status"] == "cancelado" and laudo["aprovado"] is None


def _segura_e_solta(rig, centro, executor, segundos: float, ponto_s: float = 0.3) -> None:
    """O operador manda um ponto que vale ``ponto_s`` e o servidor o repete até sumir."""
    agora = rig.clock.now()
    centro.seq_teleop = getattr(centro, "seq_teleop", 0) + 1
    centro.ponto = centro.envelope(
        "device.teleop", {"v": 0.3, "w": 0.0,
                          "valid_until": (agora + timedelta(seconds=ponto_s)).isoformat()},
        centro.seq_teleop, rig.clock, ttl=30)
    fim = time.monotonic() + segundos
    while time.monotonic() < fim:
        executor.teleop()  # o laço do Runtime
        time.sleep(0.02)


def test_homem_morto_aprova_quando_o_runtime_zera_no_valid_until(robo_real) -> None:
    rig, centro, executor = robo_real
    centro.manda("modo", {"modo": "manual"}, rig.clock)
    executor.poll()
    cid = centro.manda("ensaio", {"ensaio": "homem_morto",
                                  "params": {"timeout_s": 5, "tolerancia_s": 0.3}}, rig.clock)
    executor.poll()
    _segura_e_solta(rig, centro, executor, 1.0)
    laudo = espera_resultado(centro, cid, executor)["result"]
    assert laudo["status"] == "aprovado", laudo["criterios"]
    assert 0 <= laudo["medidas"]["atraso_max_s"] <= 0.3
    teleop = json.loads((rig.hardware.controller.teleop_path).read_text())
    assert teleop["v"] == 0.0


def test_homem_morto_reprova_runtime_que_nao_zera(robo_real, monkeypatch) -> None:
    rig, centro, executor = robo_real
    centro.manda("modo", {"modo": "manual"}, rig.clock)
    executor.poll()
    cid = centro.manda("ensaio", {"ensaio": "homem_morto",
                                  "params": {"timeout_s": 1.5, "tolerancia_s": 0.3}}, rig.clock)
    executor.poll()
    monkeypatch.setattr(executor, "_zera_teleop", lambda: None)  # o defeito: nunca zera
    _segura_e_solta(rig, centro, executor, 1.0)
    laudo = espera_resultado(centro, cid, executor)["result"]
    assert laudo["status"] == "reprovado"
    assert laudo["medidas"]["episodios"] == 0 and laudo["medidas"]["abertos"] == 1
