from __future__ import annotations

import json

import pytest

from valtexa_rover.config import Category, ConfigError, Priority, RuntimeConfig, Tier


def test_padrao_valido_e_ida_e_volta_pelo_json(tmp_path) -> None:
    cfg = RuntimeConfig().validate()
    caminho = tmp_path / "rover.json"
    caminho.write_text(json.dumps(cfg.to_dict()))
    assert RuntimeConfig.load(caminho) == cfg


def test_politica_muda_pelo_arquivo_sem_codigo(tmp_path) -> None:
    caminho = tmp_path / "rover.json"
    caminho.write_text(json.dumps({
        "retencao": {"rules": {"NORMAL": {"tier": "METADADOS", "retention_days": 1}},
                     "normal_sample_every": 50},
        "buffer": {"pre_frames": 5, "post_frames": 2, "priority": 2},
        "sync": {"chunk_size": 65536},
    }))
    cfg = RuntimeConfig.load(caminho)
    regra = cfg.retencao.rules[Category.NORMAL]
    assert regra.tier is Tier.METADADOS and regra.retention_days == 1
    assert regra.priority is Priority.NORMAL_COMPACTADA  # o que não veio fica o padrão
    assert cfg.retencao.normal_sample_every == 50
    assert cfg.buffer.pre_frames == 5 and cfg.buffer.priority is Priority.DOENCA_PRAGA
    assert cfg.sync.chunk_size == 65536


@pytest.mark.parametrize(
    ("conteudo", "trecho"),
    [
        ({"retencao_": {}}, "desconhecida"),
        ({"buffer": {"antes": 3}}, "desconhecida"),
        ({"retencao": {"rules": {"PRAGA": {}}}}, "categoria desconhecida"),
        ({"retencao": {"rules": {"NORMAL": {"tier": "REDUZIDA", "max_side": 4}}}}, "max_side"),
        ({"classificacao": {"anomaly_below": 0.9, "low_confidence_below": 0.5}}, "anomaly_below"),
        ({"sync": {"chunk_size": 10}}, "chunk_size"),
        ({"retencao": {"rules": {"NORMAL": {"tier": "TUDO"}}}}, "TUDO"),
    ],
)
def test_configuracao_invalida_e_erro_e_nao_silencio(tmp_path, conteudo, trecho) -> None:
    caminho = tmp_path / "rover.json"
    caminho.write_text(json.dumps(conteudo))
    with pytest.raises(ConfigError, match=trecho):
        RuntimeConfig.load(caminho)


def test_baixa_confianca_nunca_rebaixa_a_categoria() -> None:
    cfg = RuntimeConfig()
    regra = cfg.retencao.rule_for(Category.POSSIVEL_DOENCA, low_confidence=True)
    assert regra.tier is Tier.ORIGINAL
    assert regra.priority is Priority.DOENCA_PRAGA
    assert regra.preserve_sequence
    normal = cfg.retencao.rule_for(Category.NORMAL, low_confidence=True)
    assert normal.tier is Tier.ORIGINAL  # NORMAL duvidoso: original para validação
