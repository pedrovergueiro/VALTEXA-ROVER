"""O que o técnico roda na Jetson: provisionar o robô e preparar o modelo para o OTA."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest
from conftest import build_tiny_onnx

from valtexa_rover import cli
from valtexa_rover.runtime.client import DeviceIdentity, provision

PERFIL = {
    "version": 1, "crop": "banana", "healthy_label": "Healthy", "event_threshold": 0.8,
    "min_event_tiles": 1, "near_miss_threshold": 0.5, "anomaly_mean_confidence_below": None,
    "tile_mm": 150.0, "overlap": 0.25, "hfov_deg": 127.0, "vfov_deg": 110.0,
    "default_distance_mm": 270.0, "tile_batch": 8, "tile_probabilities": "full", "measured": {},
}


class Resposta(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def test_provisiona_gerando_a_chave_no_robo(tmp_path: Path) -> None:
    enviados = []

    def abrir(pedido, timeout):
        enviados.append((pedido.full_url, json.loads(pedido.data)))
        return Resposta(json.dumps({"device_id": "dev-9", "key_id": "k-9",
                                    "platform_public_key": "ab" * 32}).encode())

    destino = tmp_path / "identity.json"
    ident = provision("https://api.exemplo", "tok-1", "RV3-0001", destino, abrir=abrir)
    url, corpo = enviados[0]
    assert url == "https://api.exemplo/v1/device/provision"
    assert corpo["serial_number"] == "RV3-0001" and "private_key" not in json.dumps(corpo)
    assert DeviceIdentity.load(destino).private_key == ident.private_key
    with pytest.raises(FileExistsError):  # nunca troca a chave que a plataforma conhece
        provision("https://api.exemplo", "tok-1", "RV3-0001", destino, abrir=abrir)
    with pytest.raises(ValueError, match="HTTPS"):
        provision("http://api.exemplo", "tok-1", "RV3-0001", tmp_path / "x.json", abrir=abrir)


def test_provisionar_sem_token_nao_tenta(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("VALTEXA_TOKEN_PROVISIONAMENTO", raising=False)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert cli.main(["provisionar", "--url", "https://x", "--serial", "s",
                     "--identidade", str(tmp_path / "i.json")]) == 2
    assert "sem token" in capsys.readouterr().err


def test_preparar_modelo_confere_mede_e_copia_pelo_sha(tmp_path: Path, capsys) -> None:
    onnx = tmp_path / "origem" / "banana_foliar.onnx"
    sha = build_tiny_onnx(onnx)
    assert cli.main(["preparar-modelo", str(onnx), "--dados", str(tmp_path / "d")]) == 1
    assert ".edge.json não existe" in capsys.readouterr().err  # sem perfil, não vai

    onnx.with_suffix(".edge.json").write_text(json.dumps(PERFIL), encoding="utf-8")
    assert cli.main(["preparar-modelo", str(onnx), "--dados", str(tmp_path / "d"),
                     "--quadros", "1"]) == 0
    saida = json.loads(capsys.readouterr().out)
    assert saida["sha256"] == sha and saida["profile"] and saida["model_key"] == "banana_foliar"
    assert saida["medicao"]["recortes"] > 20 and saida["medicao"]["ms_por_quadro"] > 0
    staging = tmp_path / "d" / "ota" / "models"
    assert {p.name for p in staging.iterdir()} == {
        f"{sha}.onnx", f"{sha}.labels.json", f"{sha}.edge.json", f"{sha}.manifest.json"}
    # O hash impresso é o do manifesto gravado: é ele que vai para o registry.
    manifesto = (staging / f"{sha}.manifest.json").read_bytes()
    assert saida["extras_manifest_sha256"] == hashlib.sha256(manifesto).hexdigest()
    assert set(json.loads(manifesto)["arquivos"]) == {".labels.json", ".edge.json"}


def test_preparar_modelo_recusa_perfil_incoerente(tmp_path: Path, capsys) -> None:
    onnx = tmp_path / "m.onnx"
    build_tiny_onnx(onnx)
    onnx.with_suffix(".edge.json").write_text(json.dumps({**PERFIL, "healthy_label": "Sadia"}))
    assert cli.main(["preparar-modelo", str(onnx), "--dados", str(tmp_path / "d")]) == 1
    assert "rótulo sadio" in capsys.readouterr().err
    assert not list((tmp_path / "d" / "ota" / "models").iterdir())  # nada pela metade


def hal_com_camera_quebrada(*, dados: Path, url: str, clock):
    from valtexa_rover.hal.interfaces import HardwareUnavailableError
    from valtexa_rover.sim import hal_de_demonstracao

    hardware = hal_de_demonstracao(dados=dados, url=url, clock=clock)

    class Quebrada:
        camera_id = "esq-450"

        def capture(self):
            raise HardwareUnavailableError("OAK-D esq-450: sem resposta em 192.168.10.11")

    hardware.cameras.append(Quebrada())
    return hardware


def test_diagnostico_le_cada_peca_e_aponta_a_que_falhou(tmp_path: Path, capsys) -> None:
    assert cli.main(["diagnostico", "--dados", str(tmp_path / "a"), "--hal",
                     "valtexa_rover.sim:hal_de_demonstracao", "--salvar",
                     str(tmp_path / "quadros")]) == 0
    saida = json.loads(capsys.readouterr().out)
    assert saida["ok"] and {"gnss", "acelerador", "disco", "navegacao"} <= set(saida["pecas"])
    assert len(list((tmp_path / "quadros").glob("*.jpg"))) == sum(
        k.startswith("camera:") for k in saida["pecas"])

    assert cli.main(["diagnostico", "--dados", str(tmp_path / "b"), "--hal",
                     "test_instalacao:hal_com_camera_quebrada"]) == 1
    saida = json.loads(capsys.readouterr().out)
    assert not saida["pecas"]["camera:esq-450"]["ok"]
    assert "192.168.10.11" in saida["pecas"]["camera:esq-450"]["erro"]
    assert saida["pecas"]["gnss"]["ok"]  # uma peça ruim não esconde as outras


def test_diagnostico_sem_hardware_json(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.delenv("VALTEXA_HARDWARE", raising=False)
    assert cli.main(["diagnostico", "--dados", str(tmp_path)]) == 1
    assert "hardware.json" in json.loads(capsys.readouterr().out)["montagem"]
