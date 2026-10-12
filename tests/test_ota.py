"""OTA Manager: assinatura, download com retomada, instalação lado a lado, saúde e rollback.

A nuvem aqui é um DUBLÊ (assina com uma chave OTA gerada no teste). O teste
contra o SaaS de verdade está em ``saas/tests/test_ota_integracao.py``.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from datetime import timedelta
from pathlib import Path

import pytest
from conftest import LABELS, build_tiny_onnx, designation
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from valtexa_rover import launcher
from valtexa_rover.hal.interfaces import HttpResponse
from valtexa_rover.runtime.client import DeviceIdentity
from valtexa_rover.runtime.ota import canonical_ota


def tar_de(arquivos: dict[str, bytes]) -> bytes:
    saida = io.BytesIO()
    with tarfile.open(fileobj=saida, mode="w") as tar:
        for nome, dados in arquivos.items():
            info = tarfile.TarInfo(nome)
            info.size = len(dados)
            tar.addfile(info, io.BytesIO(dados))
    return saida.getvalue()


class FakeOta:
    """DUBLÊ da plataforma para o OTA: assina envelopes, serve pedaços, registra estados."""

    def __init__(self, cloud, clock) -> None:
        self.cloud = cloud
        self.clock = clock
        self.key = Ed25519PrivateKey.generate()
        self.public = self.key.public_key().public_bytes_raw().hex()
        self.deployments: dict[str, dict] = {}
        self.status: dict[str, list[str]] = {}
        self.served: list[tuple[str, int, int]] = []
        self.corrupt_bytes = False

    def publish(self, dep_id: str, kind: str, version: str, arquivos: dict[str, bytes],
                *, device_id: str = "dev-1", model: dict | None = None,
                expires_in: timedelta = timedelta(days=1), sign_with=None) -> None:
        dados = tar_de(arquivos)
        manifesto = {"kind": kind, "version": version, "sha256": hashlib.sha256(dados).hexdigest(),
                     "size_bytes": len(dados), "model": model,
                     "files": [{"path": n, "sha256": hashlib.sha256(d).hexdigest(),
                                "size_bytes": len(d)} for n, d in sorted(arquivos.items())]}
        payload = {"deployment_id": dep_id, "device_id": device_id, "package_id": "p-" + dep_id,
                   "manifest": manifesto, "issued_at": self.clock.now().isoformat(),
                   "expires_at": (self.clock.now() + expires_in).isoformat()}
        chave = sign_with or self.key
        self.deployments[dep_id] = {
            "data": dados, "state": "pendente",
            "envelope": {"payload": payload,
                         "signature": chave.sign(canonical_ota(payload)).hex()},
        }

    def __call__(self, method: str, path: str, body: bytes, headers: dict) -> HttpResponse:
        rota, _, query = path.partition("?")
        partes = rota.strip("/").split("/")
        if rota == "/v1/device/ota":
            itens = [{"deployment_id": k, "state": d["state"], "envelope": d["envelope"]}
                     for k, d in self.deployments.items()
                     if d["state"] not in ("aplicada", "revertida", "falhou", "recusada")]
            return HttpResponse(200, json.dumps(itens).encode())
        if partes[:3] == ["v1", "device", "ota"] and partes[-1] == "package":
            q = dict(p.split("=") for p in query.split("&"))
            inicio, tamanho = int(q["offset"]), int(q["length"])
            dados = self.deployments[partes[3]]["data"]
            if self.corrupt_bytes:
                dados = dados[:-1] + bytes([dados[-1] ^ 0xFF])
            self.served.append((partes[3], inicio, tamanho))
            return HttpResponse(200, dados[inicio:inicio + tamanho])
        if partes[:3] == ["v1", "device", "ota"] and partes[-1] == "status":
            corpo = json.loads(body)
            self.deployments[partes[3]]["state"] = corpo["state"]
            self.status.setdefault(partes[3], []).append(corpo["state"])
            return HttpResponse(200, b"{}")
        return self.cloud(method, path, body, headers)


@pytest.fixture
def ota_rig(rig_factory, clock):
    def fabrica(sequence=("verde",), **kwargs):
        rig = rig_factory(list(sequence), auto_sync=False, **kwargs)
        fake = FakeOta(rig.cloud, clock)
        rig.network.handler = fake
        rig.identity = DeviceIdentity(rig.identity.device_id, rig.identity.key_id,
                                      rig.identity.private_key, ota_public_key=fake.public)
        rig.core = rig.new_core()
        rig.core.ota.chunk_size = 2048
        rig.fake = fake
        return rig
    return fabrica


def _novo_modelo(tmp_path: Path, k: float = 6.0) -> tuple[bytes, str]:
    caminho = tmp_path / "novo" / "m.onnx"
    sha = build_tiny_onnx(caminho, k=k)
    return caminho.read_bytes(), sha


def _tipos(rig) -> list[str]:
    return [r["type"] for r in rig.core.store.query("SELECT type FROM records ORDER BY rowid")]


def test_modelo_assinado_instala_ao_lado_e_o_registry_decide(ota_rig, tmp_path) -> None:
    rig = ota_rig()
    dados, sha = _novo_modelo(tmp_path)
    labels = json.dumps(LABELS).encode()
    rig.fake.publish("d1", "model", "2.5.0", {f"{sha}.onnx": dados, f"{sha}.labels.json": labels},
                     model={"model_key": "coffee-foliar", "version": "2.5.0",
                            "artifact_sha256": sha, "artifact_kind": "otimizado",
                            "dnaty_version": "2.2.1"})
    rel = rig.core.ota.step()
    assert rel.applied == ["d1"], rel
    assert rig.fake.status["d1"][-4:] == ["verificando", "instalando", "verificando_saude",
                                          "aplicada"]
    assert (rig.repo / f"{sha}.onnx").exists(), "ao lado dos modelos que já existiam"
    assert (rig.repo / f"{rig.sha}.onnx").exists()
    assert rig.core.ota.current()["model"]["version"] == "2.5.0"
    # O robô só troca de modelo quando o registry designa o novo — e reconfere o hash.
    rig.cloud.designations[:] = [designation("talhao-1", sha, version="2.5.0")]
    rig.core.start_mission("m-1", "talhao-1")
    assert rig.core.engine.package.sha256 == sha
    assert {"OTA_DOWNLOADED", "OTA_INSTALLED", "OTA_APPLIED", "MODEL_UPDATED"} <= set(_tipos(rig))
    # Reenvio do mesmo destino (replay) não reinstala nada.
    rig.fake.deployments["d1"]["state"] = "pendente"
    assert rig.core.ota.step().applied == []


@pytest.mark.parametrize("caso", ["assinatura", "outro_robo", "vencido", "bytes_adulterados"])
def test_pacote_nao_confiavel_e_recusado_e_nada_muda(ota_rig, tmp_path, caso) -> None:
    rig = ota_rig()
    dados, sha = _novo_modelo(tmp_path)
    extra = {}
    if caso == "assinatura":
        extra["sign_with"] = Ed25519PrivateKey.generate()  # chave de outra pessoa
    elif caso == "outro_robo":
        extra["device_id"] = "dev-2"
    elif caso == "vencido":
        extra["expires_in"] = timedelta(seconds=-1)
    rig.fake.publish("d1", "model", "2.5.0", {f"{sha}.onnx": dados},
                     model={"artifact_sha256": sha}, **extra)
    rig.fake.corrupt_bytes = caso == "bytes_adulterados"
    rel = rig.core.ota.step()
    assert rel.rejected == ["d1"]
    assert rig.fake.status["d1"][-1] == "recusada"
    assert not (rig.repo / f"{sha}.onnx").exists()
    assert "model" not in rig.core.ota.current()
    assert "OTA_REJECTED" in _tipos(rig)
    assert not list(rig.core.ota.staging.iterdir()), "nada fica no staging"


def test_download_interrompido_retoma_do_byte_em_que_parou(ota_rig, tmp_path) -> None:
    rig = ota_rig()
    dados, sha = _novo_modelo(tmp_path)
    rig.fake.publish("d1", "model", "2.5.0", {f"{sha}.onnx": dados,
                                               f"{sha}.labels.json": json.dumps(LABELS).encode()},
                     model={"artifact_sha256": sha, "model_key": "coffee-foliar",
                            "version": "2.5.0"})
    total = rig.fake.deployments["d1"]["envelope"]["payload"]["manifest"]["size_bytes"]
    original = rig.fake.__call__

    def cai_no_meio(method, path, body, headers):
        resposta = original(method, path, body, headers)
        if "/package" in path and len(rig.fake.served) == 2:
            rig.network.set_online(False)  # a rede cai depois do segundo pedaço
        return resposta

    rig.network.handler = cai_no_meio
    assert rig.core.ota.step().deferred == "offline"
    parcial = rig.core.ota.staging / "d1.tar.part"
    assert parcial.stat().st_size == 2 * 2048
    rig.core = rig.new_core()  # reinício do robô no meio do download
    rig.core.ota.chunk_size = 2048
    rig.network.set_online(True)
    assert rig.core.ota.step().applied == ["d1"]
    inicios = [inicio for _d, inicio, _t in rig.fake.served]
    assert inicios == sorted(set(inicios)), "cada pedaço pedido uma vez só"
    assert sum(t for _d, _i, t in rig.fake.served) == total
    assert rig.fake.status["d1"][-1] == "aplicada"


def test_modelo_que_nao_carrega_falha_e_o_anterior_continua(ota_rig) -> None:
    rig = ota_rig()
    rig.core.start_mission("m-0", "talhao-1")
    rig.run(1)
    rig.core.complete_mission()
    lixo = b"isto nao e um onnx"
    sha = hashlib.sha256(lixo).hexdigest()
    rig.fake.publish("d1", "model", "9.9.9", {f"{sha}.onnx": lixo},
                     model={"artifact_sha256": sha})
    rel = rig.core.ota.step()
    assert rel.failed == ["d1"] and rig.fake.status["d1"][-1] == "falhou"
    assert not (rig.repo / f"{sha}.onnx").exists()
    assert rig.core.engine.package.sha256 == rig.sha  # o atual nem foi tocado
    assert "OTA_FAILED" in _tipos(rig)


def test_configuracao_em_teste_confirma_no_boot(ota_rig) -> None:
    rig = ota_rig()
    nova = {"sync": {"chunk_size": 65536}}
    rig.fake.publish("c1", "config", "cfg-2", {"rover.json": json.dumps(nova).encode()})
    assert rig.core.ota.step().applied == ["c1"]
    caminho = rig.core.ota.config_path
    assert json.loads(caminho.read_text()) == nova
    assert rig.core.ota.current()["config"]["state"] == "teste"
    assert rig.fake.status["c1"][-1] == "verificando_saude"  # ainda não confirmou
    revertidos, _ = launcher.prepare(rig.data_dir)  # o lançador conta o boot
    assert revertidos == []
    rig.core = rig.new_core()  # o Runtime subiu: confirma
    assert rig.core.ota.current()["config"]["state"] == "confirmado"
    assert rig.fake.status["c1"][-1] == "aplicada"


def test_configuracao_invalida_nao_entra(ota_rig) -> None:
    rig = ota_rig()
    rig.fake.publish("c1", "config", "cfg-2",
                     {"rover.json": json.dumps({"sync": {"chunk_sise": 1}}).encode()})
    assert rig.core.ota.step().failed == ["c1"]
    assert not rig.core.ota.config_path.exists()
    assert "config" not in rig.core.ota.current()


def _runtime(versao: str, quebrado: bool = False) -> dict[str, bytes]:
    init = f'RUNTIME_VERSION = "{versao}"\nEDGE_AI_VERSION = "0.1.0"\n'
    arquivos = {"valtexa_rover/__init__.py": init.encode()}
    if quebrado:
        arquivos["valtexa_rover/core.py"] = b"def quebrado(:\n"
    return arquivos


def test_runtime_novo_que_nao_confirma_volta_sozinho(ota_rig) -> None:
    rig = ota_rig()
    rig.fake.publish("r1", "runtime", "0.2.0", _runtime("0.2.0"))
    assert rig.core.ota.step().applied == ["r1"]
    atual = rig.core.ota.current()["runtime"]
    assert atual["state"] == "teste" and atual["previous"] is None
    slot = Path(atual["path"])
    assert (slot / "valtexa_rover" / "__init__.py").exists()
    # O Runtime novo não chega a confirmar: o lançador conta três boots seguidos
    # (limite padrão: 2). Nos dois primeiros ele ainda sobe o novo.
    rig.core.close()
    for _ in range(2):
        revertidos, ambiente = launcher.prepare(rig.data_dir)
        assert revertidos == [] and str(slot) in ambiente["PYTHONPATH"]
    revertidos, ambiente = launcher.prepare(rig.data_dir)
    assert revertidos == ["runtime"] and "PYTHONPATH" not in ambiente
    rig.core = rig.new_core()  # o anterior sobe; o informe guardado sai no próximo passo
    assert "runtime" not in rig.core.ota.current()
    rig.core.step()
    assert rig.fake.status["r1"][-1] == "revertida"
    assert "OTA_ROLLED_BACK" in _tipos(rig)


@pytest.mark.parametrize("quebra", ["versao_errada", "nao_compila"])
def test_runtime_que_nao_passa_na_saude_nao_e_trocado(ota_rig, quebra) -> None:
    rig = ota_rig()
    arquivos = _runtime("0.1.9") if quebra == "versao_errada" else _runtime("0.2.0", True)
    rig.fake.publish("r1", "runtime", "0.2.0", arquivos)
    assert rig.core.ota.step().failed == ["r1"]
    assert "runtime" not in rig.core.ota.current()
    assert not (rig.core.ota.slots / "runtime" / "0.2.0").exists()


def test_nada_se_instala_com_missao_em_execucao(ota_rig) -> None:
    rig = ota_rig()
    rig.fake.publish("c1", "config", "cfg-2", {"rover.json": b"{}"})
    rig.core.start_mission("m-1", "talhao-1")
    rel = rig.core.ota.step()
    assert rel.applied == [] and "missão" in rel.deferred
    assert rig.core.step().ota is None  # o laço nem pergunta durante a missão
    rig.core.complete_mission()
    assert rig.core.ota.step().applied == ["c1"]
