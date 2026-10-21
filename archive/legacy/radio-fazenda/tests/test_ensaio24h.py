"""Ensaio de 24 h SIMULADO do enlace (Parte 13, item 5), na versão de 40 min da suíte.

O mesmo código do ``python -m valtexa_rover.enlace.ensaio24h`` (24 h, ~20 min de máquina;
números em ``docs/COMUNICACAO_FAZENDA.md``), com todas as falhas do dia comprimidas: Starlink
caindo (noite e chuva), queda de energia da base, repetidora morrendo com o Rover na sombra
do morro e o rádio oscilando ao longo da rua.
"""

from __future__ import annotations

from valtexa_rover.enlace.ensaio24h import Ensaio24h, Perfil


def test_dia_comprimido_zero_perda_e_numeros_dentro_do_esperado(tmp_path) -> None:
    rel = Ensaio24h(tmp_path, Perfil.minimo()).roda()
    assert rel["simulado"] is True
    nuvem = rel["nuvem"]
    assert nuvem["aceitos"] == nuvem["registros_criados"] > 1000  # P2/P3 todos na nuvem
    assert nuvem["perdidos_p2"] == nuvem["perdidos_p3"] == nuvem["pendentes_no_rover"] == 0
    assert rel["p4"]["recebidos"] == rel["p4"]["enviados"] > 20
    assert rel["p0_p1_nao_entregues"]["perdidos_com_caminho_bom"] == 0
    assert rel["zero_perda_p0_p3"] is True
    eventos = {e["evento"] for e in rel["eventos"]}
    assert {"starlink cai", "base sem energia", "repetidora morre"} <= eventos
    # P1 rápido quando há caminho; troca de rota pela queda medida em ~1 s (calado = suspeito)
    assert rel["latencia_p1"]["p99_ms"] < 1000
    assert rel["tempo_de_troca_de_rota_s"]["p50"] < 2.0
    # o rádio que oscila ao longo da rua não faz o lote ficar pulando de enlace
    assert rel["trocas_de_enlace_com_a_base"] < 30
    assert set(rel["interrupcoes_p1_s"]) <= {"episodios", "repetidora_morta", "troca",
                                             "base_sem_energia"}
