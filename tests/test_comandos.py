"""Centro de comando, do lado do robô: busca, confere a assinatura, executa, responde."""

from __future__ import annotations

import base64
import io
import json
import secrets
from datetime import timedelta

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from PIL import Image

from valtexa_rover.hal.interfaces import HttpResponse
from valtexa_rover.hal.rover_v3 import NavegacaoPorArquivo
from valtexa_rover.runtime.client import DeviceIdentity
from valtexa_rover.runtime.comandos import CommandExecutor
from valtexa_rover.runtime.mission import canonical_command


class CentroDeComando:
    """O lado da plataforma, em memória: assina como ``valtexa.security.signing``."""

    def __init__(self, device_id: str, nuvem) -> None:
        self.chave = Ed25519PrivateKey.generate()
        self.publica = self.chave.public_key().public_bytes_raw().hex()
        self.device_id = device_id
        self.nuvem = nuvem
        self.fila: list[dict] = []
        self.resultados: dict[str, dict] = {}
        self.ponto: dict | None = None
        self.seq = 0

    def envelope(self, command: str, body: dict, sequence: int, clock, ttl: int = 120) -> dict:
        agora = clock.now()
        payload = {"command": command, "device_id": self.device_id, "sequence": sequence,
                   "nonce": secrets.token_hex(8), "issued_at": agora.isoformat(),
                   "expires_at": (agora + timedelta(seconds=ttl)).isoformat(), "body": body}
        return {"payload": payload, "signature": self.chave.sign(canonical_command(payload)).hex()}

    def manda(self, kind: str, params: dict, clock) -> str:
        self.seq += 1
        cid = f"c-{self.seq}"
        self.fila.append(self.envelope(f"device.{kind}",
                                       {"command_id": cid, "kind": kind, "params": params},
                                       self.seq, clock))
        return cid

    def __call__(self, method, path, body, headers):
        if path == "/v1/device/commands":
            saida, self.fila = self.fila, []
            return HttpResponse(200, json.dumps(saida).encode())
        if path.startswith("/v1/device/commands/") and path.endswith("/result"):
            self.resultados[path.split("/")[4]] = json.loads(body)
            return HttpResponse(200, b"{}")
        if path == "/v1/device/teleop":
            if self.ponto is None:
                return HttpResponse(204, b"")
            return HttpResponse(200, json.dumps(self.ponto).encode())
        return self.nuvem(method, path, body, headers)


@pytest.fixture
def robo(rig_factory, tmp_path):
    rig = rig_factory(["verde", "vermelho"])
    centro = CentroDeComando(rig.identity.device_id, rig.cloud)
    rig.identity = DeviceIdentity(rig.identity.device_id, rig.identity.key_id,
                                  rig.identity.private_key, centro.publica)
    rig.network.handler = centro  # a "rede" do robô agora fala com o centro de comando
    rig.hardware.controller = NavegacaoPorArquivo(tmp_path / "run" / "navegacao.json",
                                                  tmp_path / "run" / "parar.json",
                                                  clock=rig.clock)
    rig.core = rig.new_core()
    return rig, centro, CommandExecutor(rig.core, tmp_path / "rover.json")


def test_diagnostico_foto_e_modo(robo, tmp_path) -> None:
    rig, centro, executor = robo
    diag = centro.manda("diagnostico", {"pecas": ["cameras", "gnss", "disco"], "fotos": True},
                        rig.clock)
    foto = centro.manda("foto", {"camera_id": "cam-0", "lado": 200}, rig.clock)
    modo = centro.manda("modo", {"modo": "pausado"}, rig.clock)
    assert executor.poll() == 3

    laudo = centro.resultados[diag]
    assert laudo["ok"] and set(laudo["result"]["pecas"]) == {"camera:cam-0", "gnss", "disco"}
    assert "jpeg_base64" in laudo["result"]["pecas"]["camera:cam-0"]
    miniatura = base64.b64decode(centro.resultados[foto]["result"]["jpeg_base64"])
    assert max(Image.open(io.BytesIO(miniatura)).size) <= 200
    assert centro.resultados[modo] == {"ok": True, "result": {"modo": "pausado"}, "error": None}
    controle = json.loads((tmp_path / "run" / "controle.json").read_text())
    assert controle["modo"] == "pausado"


def test_assinatura_falsa_e_repeticao_sao_recusadas(robo) -> None:
    rig, centro, executor = robo
    centro.manda("modo", {"modo": "manual"}, rig.clock)
    falso = dict(centro.fila[-1])
    assert executor.poll() == 1
    centro.fila.append(falso)  # o mesmo envelope de novo: sequência repetida
    outro = Ed25519PrivateKey.generate()
    forjado = centro.envelope("device.parar", {"command_id": "x", "kind": "parar",
                                               "params": {}}, 99, rig.clock)
    forjado["signature"] = outro.sign(canonical_command(forjado["payload"])).hex()
    centro.fila.append(forjado)
    assert executor.poll() == 0 and "x" not in centro.resultados


def test_controle_remoto_com_homem_morto(robo, tmp_path) -> None:
    rig, centro, executor = robo
    centro.manda("modo", {"modo": "manual"}, rig.clock)
    executor.poll()
    valido = (rig.clock.now() + timedelta(seconds=1)).isoformat()
    centro.ponto = centro.envelope("device.teleop", {"v": 0.3, "w": -0.1,
                                                     "valid_until": valido}, 7, rig.clock, ttl=2)
    assert executor.teleop()["v"] == 0.3
    ponto = json.loads((tmp_path / "run" / "teleop.json").read_text())
    assert (ponto["v"], ponto["w"], ponto["seq"]) == (0.3, -0.1, 7)

    centro.ponto = None  # a equipe soltou o comando: a navegação recebe zero
    executor.teleop()
    assert json.loads((tmp_path / "run" / "teleop.json").read_text())["v"] == 0.0

    centro.manda("modo", {"modo": "autonomo"}, rig.clock)
    executor.poll()
    centro.ponto = centro.envelope("device.teleop", {"v": 0.4, "w": 0.0,
                                                     "valid_until": valido}, 8, rig.clock)
    assert executor.teleop() is None  # fora do manual, o controle remoto não age


def test_config_valida_troca_e_pede_reinicio(robo, tmp_path) -> None:
    rig, centro, executor = robo
    ruim = centro.manda("config", {"rover_json": {"retencao": {"normal_sample_every": -3}}},
                        rig.clock)
    executor.poll()
    assert not centro.resultados[ruim]["ok"] and not executor.restart_requested
    boa = centro.manda("config", {"rover_json": {"retencao": {"normal_sample_every": 1}}},
                       rig.clock)
    executor.poll()
    assert centro.resultados[boa]["ok"] and executor.restart_requested
    gravada = json.loads((tmp_path / "rover.json").read_text())
    assert gravada["retencao"]["normal_sample_every"] == 1  # o modo coleta
    assert "sync" in gravada  # o resto da configuração continua lá


def test_comando_desconhecido_e_parar(robo, tmp_path) -> None:
    rig, centro, executor = robo
    estranho = centro.manda("voar", {}, rig.clock)
    parar = centro.manda("parar", {"motivo": "pessoa na rua"}, rig.clock)
    executor.poll()
    assert "desconhecido" in centro.resultados[estranho]["error"]
    assert centro.resultados[parar]["ok"]
    assert json.loads((tmp_path / "run" / "parar.json").read_text())["reason"] == "pessoa na rua"
