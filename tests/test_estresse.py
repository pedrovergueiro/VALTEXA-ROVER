"""Estresse (seção 42) em tamanho de CI. Os números grandes saem de
``python -m valtexa_rover.sim.estresse --passos 50000`` (ver docs/ESTRESSE.md).

``rodar`` já confere as invariantes (cada registro uma vez na nuvem, missão sem
quadro perdido nem contado duas vezes, fila vazia) e levanta se alguma quebrar.
"""

from __future__ import annotations

from valtexa_rover.sim.estresse import rodar


def test_rede_intermitente_backend_fora_reinicios_e_camera_fora(tmp_path) -> None:
    num = rodar(tmp_path / "r", passos=400, queda_a_cada=60, queda_dura=25,
                reinicio_a_cada=150, rajada_5xx_a_cada=120)
    assert num.quadros > 700 and num.falhas_de_camera > 0
    assert num.reinicios == 2 and num.passos_sem_rede >= 150
    assert num.respostas_5xx_injetadas >= 24
    assert num.registros_na_nuvem == num.registros_locais
    assert num.originais_de_evento_na_nuvem == num.originais_de_evento > 0
    assert num.falhou_na_fila == 0


def test_disco_cheio_libera_espaco_sem_perder_original_de_evento(tmp_path) -> None:
    num = rodar(tmp_path / "r", passos=300, queda_a_cada=100, queda_dura=80,
                reinicio_a_cada=0, rajada_5xx_a_cada=0, cota_mb=1.5)
    assert num.limpezas_por_disco_cheio > 0, "a cota pequena precisa forçar a limpeza"
    # O que a retenção pode apagar sob pressão: o que já subiu e as reduzidas NORMAL.
    # Original de evento que não subiu, nunca.
    assert num.originais_de_evento_na_nuvem == num.originais_de_evento > 0
    assert num.registros_na_nuvem == num.registros_locais
