"""As prioridades da missão (Nível 4): o Rover gasta a percepção ativa onde vale mais — e a
decisão e a evidência que sobem não mudam."""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from test_percepcao import AMBIGUA, cena, motor, primeira, relogio  # noqa: F401

from valtexa_rover._compat import UTC
from valtexa_rover.config import ActivePerceptionConfig, ConfigError
from valtexa_rover.runtime import prioridades
from valtexa_rover.runtime.core import RoverCore
from valtexa_rover.runtime.percepcao import ActivePerception

AGORA = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)


def _corpo(valido_ate: datetime = AGORA + timedelta(hours=6), **trecho) -> dict:
    t = {"ruas": ["G3"], "x_inicio_m": 10.0, "x_fim_m": 30.0, "prioridade": 1.0, **trecho}
    return {"prioridades": {"plano_id": "pl-1", "valido_ate": valido_ate.isoformat(),
                            "uso": "decidir a bordo", "trechos": [
                                t, {"ruas": ["G3"], "x_inicio_m": 31.0, "x_fim_m": 50.0,
                                    "prioridade": 0.4}]}}


def test_mapa_da_missao_valido_vencido_e_malformado() -> None:
    mapa = prioridades.de_missao(_corpo(), AGORA)
    assert mapa is not None and mapa.plano_id == "pl-1" and len(mapa.trechos) == 2
    assert prioridades.de_missao(_corpo(AGORA - timedelta(minutes=1)), AGORA) is None
    assert prioridades.de_missao(_corpo(prioridade=2.0), AGORA) is None
    assert prioridades.de_missao({"prioridades": {"trechos": []}}, AGORA) is None
    assert prioridades.de_missao({}, AGORA) is None and prioridades.de_missao(None, AGORA) is None


def test_prioridade_por_rua_e_eixo() -> None:
    mapa = prioridades.de_missao(_corpo(), AGORA)
    assert mapa.prioridade("G3", 20.0) == 1.0
    assert mapa.prioridade("G3", 30.5) == 1.0  # folga de 1 m: vale o maior dos dois
    assert mapa.prioridade("G3", 45.0) == 0.4
    assert mapa.prioridade("G3", 80.0) == 0.0 and mapa.prioridade("G4", 20.0) == 0.0
    assert mapa.prioridade(None, 20.0) is None and mapa.prioridade("G3", None) is None


def test_trecho_de_baixa_prioridade_nao_gasta_vista(tmp_path, relogio) -> None:  # noqa: F811
    m = motor(tmp_path)
    camera = cena(relogio, cor=AMBIGUA)
    quadro, r = primeira(m, camera)
    ap = ActivePerception(ActivePerceptionConfig(), relogio)
    d = ap.run(m, camera, quadro, r, prioridade=0.05)
    assert d.block["skipped"] == "prioridade_baixa" and d.block["views"] == []
    assert d.block["prioridade"] == 0.05 and ap.snapshot()["prioridade_baixa"] == 1
    # Só o "quanto insistir" muda: a decisão é a de uma vista, os recortes são os mesmos.
    assert d.inference.category is r.category and d.inference.tiles == r.tiles
    alta = ap.run(m, camera, *primeira(m, camera), prioridade=0.9)
    assert len(alta.block["views"]) >= 1


def test_reserva_vale_a_cada_vista(tmp_path, relogio) -> None:  # noqa: F811
    m = motor(tmp_path)
    camera = cena(relogio, cor=AMBIGUA)
    cfg = ActivePerceptionConfig(max_vistas_extras=2, max_extras_por_minuto=3,
                                 reserva_prioridade_alta=2)
    ap = ActivePerception(cfg, relogio)
    media = ap.run(m, camera, *primeira(m, camera), prioridade=0.3)
    assert len(media.block["views"]) == 1  # 3 disponíveis, piso 2: uma vista só
    assert media.block["skipped"] == "orcamento_reservado"
    nada = ap.run(m, camera, *primeira(m, camera), prioridade=0.3)
    assert nada.block["views"] == [] and nada.block["skipped"] == "orcamento_reservado"
    # Só a reserva sobrou (2 por minuto): o trecho que vale mais ainda olha de novo.
    alta = ap.run(m, camera, *primeira(m, camera), prioridade=0.95)
    assert alta.block["skipped"] is None and len(alta.block["views"]) >= 1


def test_sem_mapa_age_como_antes(tmp_path, relogio) -> None:  # noqa: F811
    m = motor(tmp_path)
    camera = cena(relogio, cor=AMBIGUA)
    d = ActivePerception(ActivePerceptionConfig(max_vistas_extras=2), relogio).run(
        m, camera, *primeira(m, camera))
    assert len(d.block["views"]) == 2 and d.block["prioridade"] is None


def test_configuracao_da_prioridade() -> None:
    with pytest.raises(ConfigError):
        ActivePerceptionConfig(prioridade_minima=0.6, prioridade_alta=0.5).validate()
    with pytest.raises(ConfigError):
        ActivePerceptionConfig(reserva_prioridade_alta=-1).validate()


def test_core_cruza_lado_da_camera_rua_e_eixo() -> None:
    estado = {"row_left": "G3", "row_right": "G4", "x_m": 20.0}
    stub = SimpleNamespace(clock=SimpleNamespace(now=lambda: AGORA),
                           _navigation=lambda: estado)
    missao = SimpleNamespace(id="m-1", payload=_corpo())
    esquerda = SimpleNamespace(side="left")
    direita = SimpleNamespace(side="right")
    assert RoverCore._prioridade(stub, missao, esquerda) == 1.0
    assert RoverCore._prioridade(stub, missao, direita) == 0.0  # G4 não tem trecho
    assert RoverCore._prioridade(stub, None, esquerda) is None
    sem_mapa = SimpleNamespace(id="m-2", payload={})
    assert RoverCore._prioridade(stub, sem_mapa, esquerda) is None
