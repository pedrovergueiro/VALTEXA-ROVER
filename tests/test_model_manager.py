"""O Rover só roda modelo do registry, com o hash conferido (seções 9 e 22)."""

from __future__ import annotations

import hashlib
import json

import pytest
from conftest import designation

from valtexa_rover.runtime.model_manager import ModelError, manifesto_dos_extras


def _eventos(rig, tipo: str) -> list:
    return rig.core.store.query("SELECT payload FROM records WHERE type = ?", (tipo,))


def test_carrega_o_designado_com_hash_conferido(rig_factory) -> None:
    rig = rig_factory(["verde"])
    pacote = rig.core.models.activate_for_field("talhao-1")
    assert pacote.sha256 == rig.sha and pacote.version == "2.4.1"
    assert pacote.artifact_kind == "otimizado" and pacote.dnaty_version == "2.2.1"
    assert rig.core.engine.loaded
    assert rig.core.store.get("active_model")["sha256"] == rig.sha
    assert len(_eventos(rig, "MODEL_UPDATED")) == 1


def test_hash_diferente_do_registry_e_recusado(rig_factory) -> None:
    rig = rig_factory(["verde"])
    adulterado = rig.repo / f"{rig.sha}.onnx"
    adulterado.write_bytes(adulterado.read_bytes() + b"\x00")
    # Robô novo: ainda sem a cópia conferida. O boot tenta pré-carregar e recusa.
    (rig.core.models.models_dir / f"{rig.sha}.onnx").unlink()
    rig.core.close()
    rig.core = rig.new_core()
    assert not rig.core.engine.loaded
    assert "não confere" in (rig.core.health.modelo_indisponivel or "")
    with pytest.raises(ModelError, match="não confere"):
        rig.core.models.activate_for_field("talhao-1")
    assert not rig.core.engine.loaded
    assert len(_eventos(rig, "MODEL_REJECTED")) >= 1
    assert not (rig.core.models.models_dir / f"{rig.sha}.onnx").exists()


def test_designacao_sem_hash_nao_carrega(rig_factory) -> None:
    rig = rig_factory(["verde"])
    rig.cloud.designations = [designation("talhao-1", None)]
    rig.core.models.refresh()
    with pytest.raises(ModelError, match="sem hash"):
        rig.core.models.activate_for_field("talhao-1")


def test_talhao_sem_modelo_ou_nao_designado(rig_factory) -> None:
    rig = rig_factory(["verde"])
    rig.cloud.designations = [{**designation("talhao-1", rig.sha), "model_key": None,
                               "version": None, "reason": "nenhum modelo atende"}]
    rig.core.models.refresh()
    with pytest.raises(ModelError, match="nenhum modelo atende"):
        rig.core.models.activate_for_field("talhao-1")
    with pytest.raises(ModelError, match="não designou"):
        rig.core.models.activate_for_field("talhao-2")


def test_arquivo_na_pasta_sem_designacao_nunca_carrega(rig_factory) -> None:
    """Modelo fora do registry: estar na pasta não basta."""
    rig = rig_factory(["verde"])
    rig.cloud.designations = [designation("talhao-1", "f" * 64)]
    rig.core.models.refresh()
    with pytest.raises(ModelError, match="não está no robô"):
        rig.core.models.activate_for_field("talhao-1")
    pacote = rig.core.engine.package  # o do boot pode seguir na memória; o "f…" nunca
    assert pacote is None or pacote.sha256 != "f" * 64
    with pytest.raises(ModelError):
        rig.core.start_mission("m-1", "talhao-1")  # e a missão não começa com ele


def test_missao_nao_comeca_sem_modelo_do_registry(rig_factory) -> None:
    rig = rig_factory(["verde"])
    rig.cloud.designations = [designation("talhao-1", None)]
    with pytest.raises(ModelError):
        rig.core.start_mission("m-1", "talhao-1")
    assert rig.core.missions.current() is None


def test_offline_usa_a_ultima_designacao_e_reconfere_o_hash(rig_factory) -> None:
    rig = rig_factory(["verde"] * 3)
    rig.core.models.refresh()  # online uma vez
    rig.network.set_online(False)
    rig.core.close()
    core = rig.new_core()  # reinício sem rede
    rig.core = core
    core.start_mission("m-1", "talhao-1")
    assert core.engine.package.sha256 == rig.sha
    rig.run(3)
    assert core.missions.current().frames_done == 3


def test_versoes_informadas_a_plataforma(rig_factory) -> None:
    rig = rig_factory(["verde"])
    rig.core.start_mission("m-1", "talhao-1")
    rig.core.sync.step()
    sessao = rig.cloud.sessions[-1]
    assert sessao["model_key"] == "coffee-foliar" and sessao["model_version"] == "2.4.1"
    assert sessao["dnaty_version"] == "2.2.1"
    assert sessao["hardware_version"] == "Rover Edge V1 (simulado)"
    assert sessao["runtime_version"] and sessao["edge_ai_version"]
    rig.core.sync.step()
    assert len(rig.cloud.sessions) == 1  # só informa de novo quando muda


def _com_manifesto(rig) -> str:
    """Grava o manifesto dos extras na origem e designa o modelo com o hash dele."""
    manifesto = manifesto_dos_extras(rig.repo / f"{rig.sha}.onnx")
    (rig.repo / f"{rig.sha}.manifest.json").write_bytes(manifesto)
    esperado = hashlib.sha256(manifesto).hexdigest()
    rig.cloud.designations = [designation("talhao-1", rig.sha, extras_manifest_sha256=esperado)]
    rig.core.models.refresh()
    for arquivo in rig.core.models.models_dir.glob(f"{rig.sha}.*"):
        arquivo.unlink()  # robô sem a cópia: busca na origem e confere tudo
    return esperado


def test_extras_conferidos_pelo_manifesto_do_registry(rig_factory) -> None:
    rig = rig_factory(["verde"])
    _com_manifesto(rig)
    rig.core.close()
    rig.core = rig.new_core()
    pacote = rig.core.models.activate_for_field("talhao-1")
    assert pacote.sha256 == rig.sha and rig.core.engine.loaded
    assert (rig.core.models.models_dir / f"{rig.sha}.manifest.json").exists()


@pytest.mark.parametrize("estrago", ["alterado", "sobrando", "faltando", "manifesto"])
def test_extra_que_nao_bate_com_o_manifesto_e_recusado(rig_factory, estrago) -> None:
    rig = rig_factory(["verde"])
    _com_manifesto(rig)
    origem = rig.repo / f"{rig.sha}.onnx"
    if estrago == "alterado":
        origem.with_suffix(".labels.json").write_text(json.dumps(["sadia", "outra"]))
    elif estrago == "sobrando":  # uma cabeça "adaptada" enfiada sem passar pelo registry
        origem.with_suffix(".conformal.json").write_text("{}")
    elif estrago == "faltando":
        origem.with_suffix(".labels.json").unlink()
    else:
        origem.with_suffix(".manifest.json").write_bytes(b'{"formato":1,"arquivos":{}}')
    rig.core.close()
    rig.core = rig.new_core()
    assert not rig.core.engine.loaded
    with pytest.raises(ModelError, match="manifesto"):
        rig.core.models.activate_for_field("talhao-1")
    assert not rig.core.engine.loaded
    recusas = [json.loads(e[0])["payload"] for e in _eventos(rig, "MODEL_REJECTED")]
    assert any(r.get("expected_manifest_sha256") for r in recusas)
    # O pacote ruim sai do robô: a próxima ativação busca de novo na origem.
    assert not list(rig.core.models.models_dir.glob(f"{rig.sha}.*"))


def test_designacao_sem_manifesto_continua_carregando(rig_factory) -> None:
    rig = rig_factory(["verde"])
    rig.core.models.activate_for_field("talhao-1")
    [evento] = _eventos(rig, "MODEL_UPDATED")
    assert json.loads(evento[0])["payload"]["extras_conferidos"] is False
