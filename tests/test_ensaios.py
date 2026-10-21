"""Ensaios automáticos na bancada SIMULADA: cada um aprova o são e reprova a falha injetada."""

from __future__ import annotations

import math
import time

import pytest

from valtexa_rover.clock import SystemClock
from valtexa_rover.ensaios import (
    CATALOGO,
    Contexto,
    ExecutorDeEnsaios,
    catalogo,
    rodar_por_nome,
)
from valtexa_rover.ensaios.calor import queda_de_clock_pct
from valtexa_rover.ensaios.energia import SensoresComoEnergia
from valtexa_rover.ensaios.estatistica import (
    inclinacao_pela_gravidade,
    percentil,
    precisao_horizontal,
)
from valtexa_rover.ensaios.simulado import CargaDeBancada, TermicaDeBancada, bancada
from valtexa_rover.hal.simulated import SimulatedSensors


def ensaio(tmp_path, clock, nome, params=None, falhas=None, cameras=8, **fontes_extra):
    hardware, fontes = bancada(tmp_path / "bancada", clock, cameras=cameras, falhas=falhas)
    fontes.update(fontes_extra)
    return rodar_por_nome(nome, params, Contexto(hardware, clock, fontes))


def criterios(laudo) -> dict[str, bool]:
    return {c["nome"]: c["ok"] for c in laudo["criterios"]}


# -- contas -----------------------------------------------------------------------------------


def test_contas_basicas() -> None:
    assert percentil([1, 2, 3, 4], 50) == 2.5
    assert percentil([5], 95) == 5
    roll, pitch = inclinacao_pela_gravidade(0.0, 9.80665 * math.sin(math.radians(10)),
                                            9.80665 * math.cos(math.radians(10)))
    assert roll == pytest.approx(10) and pitch == pytest.approx(0)
    # Quatro pontos a 1 cm do ponto conhecido, nos quatro rumos: CEP de 1 cm, sem viés.
    d = math.degrees(0.01 / 6_378_137.0)
    lats = [d, -d, 0.0, 0.0]
    lons = [0.0, 0.0, d, -d]
    prec = precisao_horizontal(lats, lons, (0.0, 0.0))
    assert prec["cep50_cm"] == pytest.approx(1.0, abs=1e-3)
    assert prec["vies_cm"] == pytest.approx(0.0, abs=1e-6)
    assert prec["drms2_cm"] == pytest.approx(2.0, abs=1e-3)
    assert queda_de_clock_pct([1000, 1000, 700, None]) == pytest.approx(30)


def test_catalogo_e_parametros(tmp_path, clock) -> None:
    nomes = {e["nome"] for e in catalogo()}
    assert nomes == {"calor", "cameras", "profundidade", "gnss", "imu", "rede", "energia",
                     "relogio", "parada", "homem_morto"}
    assert all(e["hardware_real"] for e in catalogo())  # cada um diz o que falta no real
    laudo = ensaio(tmp_path, clock, "cameras", {"fps": 999})
    assert laudo["status"] == "erro" and "fps" in laudo["erro"]
    laudo = ensaio(tmp_path, clock, "cameras", {"voar": 1})
    assert laudo["status"] == "erro" and "desconhecidos" in laudo["erro"]
    laudo = ensaio(tmp_path, clock, "profundidade", {"distancia_real_mm": 500})
    assert laudo["status"] == "erro" and "camera_id" in laudo["erro"]


# -- calor ------------------------------------------------------------------------------------


def test_calor_aprova_sem_throttling(tmp_path, clock) -> None:
    laudo = ensaio(tmp_path, clock, "calor", {"duracao_min": 3, "janela_s": 30})
    assert laudo["status"] == "aprovado", laudo["criterios"]
    assert laudo["simulado"] is True
    assert laudo["medidas"]["vazao_primeira_qps"] == pytest.approx(8.0, rel=0.05)
    assert laudo["medidas"]["temp_max_c"] == pytest.approx(48.0, abs=0.1)
    assert len(laudo["amostras"]["janelas_qps"]) == 6


def test_calor_reprova_com_throttling(tmp_path, clock) -> None:
    laudo = ensaio(tmp_path, clock, "calor",
                   {"duracao_min": 4, "janela_s": 30, "temp_max_c": 60},
                   falhas={"termica": {"aquece_c_por_min": 5, "throttle_c": 55}})
    assert laudo["status"] == "reprovado"
    ok = criterios(laudo)
    assert not ok["perda_vazao_pct"] and not ok["queda_clock_pct"] and not ok["temp_max_c"]
    assert laudo["medidas"]["perda_vazao_pct"] == pytest.approx(30, abs=1)


def test_calor_sem_modelo_diz_o_que_falta(tmp_path, clock) -> None:
    hardware, fontes = bancada(tmp_path / "b", clock)
    del fontes["carga"]
    laudo = rodar_por_nome("calor", {"duracao_min": 1}, Contexto(hardware, clock, fontes))
    assert laudo["status"] == "erro" and "modelo" in laudo["erro"]


# -- câmeras ----------------------------------------------------------------------------------


def test_cameras_oito_juntas_aprovam(tmp_path, clock) -> None:
    laudo = ensaio(tmp_path, clock, "cameras", {"duracao_s": 10, "fps": 2})
    assert laudo["status"] == "aprovado", laudo["criterios"]
    assert laudo["medidas"]["cameras"] == 8 and laudo["medidas"]["esperadas"] == 160


def test_cameras_perda_salto_e_deriva(tmp_path, clock) -> None:
    laudo = ensaio(tmp_path, clock, "cameras", {"duracao_s": 10, "fps": 2, "cameras": 4},
                   falhas={"camera:1": {"falha_em": [3, 4]}, "camera:2": {"salto_em": [5]},
                           "camera:3": {"atraso_carimbo_s": 0.4}})
    assert laudo["status"] == "reprovado"
    m = laudo["medidas"]
    assert m["perdidas"] == 3 and m["por_camera"]["oak-1"]["falhas"] == 2
    assert m["por_camera"]["oak-2"]["saltos"] == 1
    ok = criterios(laudo)
    assert not ok["perda_pct"] and not ok["deriva_p95_ms"] and not ok["gnss_p95_ms"]
    assert m["deriva_p95_ms"] == pytest.approx(400, abs=1)


def test_cameras_mais_que_o_rover_tem(tmp_path, clock) -> None:
    laudo = ensaio(tmp_path, clock, "cameras", {"cameras": 8}, cameras=2)
    assert laudo["status"] == "erro" and "tem 2" in laudo["erro"]


# -- profundidade -----------------------------------------------------------------------------


def test_profundidade_contra_trena(tmp_path, clock) -> None:
    base = {"camera_id": "oak-0", "distancia_real_mm": 500, "amostras": 5}
    assert ensaio(tmp_path, clock, "profundidade", base)["status"] == "aprovado"
    torto = ensaio(tmp_path, clock, "profundidade", base,
                   falhas={"camera": {"vies_mm": 35}})
    assert torto["status"] == "reprovado"
    assert torto["medidas"]["erro_mm"] == pytest.approx(35, abs=2)
    furado = ensaio(tmp_path, clock, "profundidade", base,
                    falhas={"camera": {"preenchimento": 0.2}})
    assert criterios(furado) == {"erro_abs_mm": True, "preenchimento": False}
    roi = ensaio(tmp_path, clock, "profundidade", {**base, "roi": [0.9, 0.9, 0.3, 0.3]})
    assert roi["status"] == "erro" and "roi" in roi["erro"]


# -- GNSS -------------------------------------------------------------------------------------


def test_gnss_fix_precisao_e_perda(tmp_path, clock) -> None:
    params = {"tempo_fix_max_s": 60, "duracao_s": 30, "lat_ref": -21.25, "lon_ref": -45.0,
              "perda_s": 40, "taxa_hz": 5}
    # O fix chega em 20 s; a precisão vai de ~20 a 50 s; a perda simulada, de 60 a 70 s.
    laudo = ensaio(tmp_path, clock, "gnss", params, falhas={"gnss": {"perdas": [(60, 70)]}})
    assert laudo["status"] == "aprovado", laudo["criterios"]
    m = laudo["medidas"]
    assert m["tempo_fix_s"] == pytest.approx(20, abs=0.5)
    assert m["cep95_cm"] < 5 and m["vies_cm"] < 1
    assert len(m["perdas"]) == 1 and m["perdas"][0]["retomada_s"] == pytest.approx(10, abs=0.5)
    assert len(laudo["amostras"]["lat"]) == 150


def test_gnss_reprova_vies_sem_fix_e_perda_nao_feita(tmp_path, clock) -> None:
    ref = {"lat_ref": -21.25, "lon_ref": -45.0, "duracao_s": 10}
    torto = ensaio(tmp_path, clock, "gnss", ref, falhas={"gnss": {"vies_cm": (8.0, 0.0)}})
    assert criterios(torto)["vies_cm"] is False
    sem_fix = ensaio(tmp_path, clock, "gnss", {"tempo_fix_max_s": 10},
                     falhas={"gnss": {"tempo_fix_s": 500}})
    assert sem_fix["status"] == "reprovado" and len(sem_fix["criterios"]) == 1
    assert "não medidas" in sem_fix["medidas"]["observacao"]
    sem_perda = ensaio(tmp_path, clock, "gnss", {"duracao_s": 5, "perda_s": 10})
    assert criterios(sem_perda)["perdas_observadas"] is False


# -- IMU --------------------------------------------------------------------------------------


def test_imu_parada_aprova_e_acha_eixos_trocados(tmp_path, clock) -> None:
    inclinada = {"duracao_s": 5, "roll_esperado_deg": 6.0, "pitch_esperado_deg": -3.0}
    falha_base = {"roll": 6.0, "pitch": -3.0}
    ok = ensaio(tmp_path, clock, "imu", inclinada, falhas={"imu": falha_base})
    assert ok["status"] == "aprovado", ok["criterios"]
    assert ok["medidas"]["gravidade_ms2"] == pytest.approx(9.80665, abs=0.01)
    assert ok["medidas"]["fonte_gyro"] == "giroscópio"
    trocado = ensaio(tmp_path, clock, "imu", inclinada,
                     falhas={"imu": {**falha_base, "eixos_trocados": True}})
    assert not criterios(trocado)["eixos_roll_deg"]
    escala = ensaio(tmp_path, clock, "imu", inclinada,
                    falhas={"imu": {**falha_base, "escala_g": 1.03}})
    assert not criterios(escala)["erro_gravidade_ms2"]
    ruidosa = ensaio(tmp_path, clock, "imu", inclinada,
                     falhas={"imu": {**falha_base, "ruido_gyro_dps": 1.0}})
    assert not criterios(ruidosa)["ruido_gyro_dps"]


def test_imu_sem_giroscopio_usa_a_derivada_e_diz(tmp_path, clock) -> None:
    laudo = ensaio(tmp_path, clock, "imu", {"duracao_s": 5},
                   falhas={"imu": {"com_gyro": False, "ruido_angulo_deg": 0.0001}})
    assert laudo["status"] == "aprovado", laudo["criterios"]
    assert "derivada" in laudo["medidas"]["fonte_gyro"]


# -- segurança --------------------------------------------------------------------------------


def test_parada_mede_a_latencia_ate_a_confirmacao(tmp_path, clock) -> None:
    laudo = ensaio(tmp_path, clock, "parada", {"tentativas": 3, "intervalo_s": 1})
    assert laudo["status"] == "aprovado", laudo["criterios"]
    assert laudo["medidas"]["latencia_p95_ms"] == pytest.approx(120, abs=15)
    assert laudo["medidas"]["modo_final"] == "parado"
    lenta = ensaio(tmp_path, clock, "parada", {"tentativas": 2},
                   falhas={"navegacao": {"latencias": [0.1, 0.9]}})
    assert criterios(lenta) == {"confirmadas_pct": True, "latencia_p95_ms": False}
    surda = ensaio(tmp_path, clock, "parada", {"tentativas": 2, "timeout_s": 1},
                   falhas={"navegacao": {"confirma": False}})
    assert surda["status"] == "reprovado" and surda["medidas"]["confirmadas"] == 0


def test_parada_exigindo_movimento_nao_conta_robo_parado(tmp_path, clock) -> None:
    laudo = ensaio(tmp_path, clock, "parada",
                   {"tentativas": 1, "exigir_movimento": True, "espera_movimento_s": 2})
    assert laudo["status"] == "reprovado" and laudo["medidas"]["sem_movimento"] == 1


# -- rede e energia ---------------------------------------------------------------------------


def test_rede_rtt_perda_e_vazao(tmp_path, clock) -> None:
    ok = ensaio(tmp_path, clock, "rede", {"pings": 10, "carga_kb": 64})
    assert ok["status"] == "aprovado", ok["criterios"]
    assert ok["medidas"]["rtt_p95_ms"] == pytest.approx(80, abs=1)
    ruim = ensaio(tmp_path, clock, "rede", {"pings": 10, "carga_kb": 64},
                  falhas={"rede": {"perde_em": {0, 1}, "kbps": 100}})
    assert criterios(ruim) == {"perda_pct": False, "rtt_p95_ms": True,
                               "vazao_subida_kbps": False, "vazao_descida_kbps": False}


def test_energia_sem_bms_nao_inventa_numero(tmp_path, clock) -> None:
    laudo = ensaio(tmp_path, clock, "energia", {"duracao_min": 1})
    assert laudo["status"] == "nao_avaliado" and laudo["aprovado"] is None
    assert "aguarda protocolo do BMS/MCU" in laudo["observacao"]
    com_fonte = ensaio(tmp_path, clock, "energia", {"duracao_min": 30, "amostra_s": 60},
                       energia=SensoresComoEnergia(SimulatedSensors(clock)))
    assert com_fonte["status"] == "aprovado"  # 8 %/h simulado → 12,5 h
    assert com_fonte["medidas"]["autonomia_estimada_h"] == pytest.approx(12.5, rel=0.02)


# -- execução em thread: progresso e cancelamento ---------------------------------------------


def test_executor_progresso_cancelamento_e_um_por_vez(tmp_path) -> None:
    relogio = SystemClock()
    hardware, _ = bancada(tmp_path / "b", relogio, cameras=1)
    termica = TermicaDeBancada(relogio)
    progresso, laudos = [], {}
    executor = ExecutorDeEnsaios(
        hardware, relogio, {"carga": CargaDeBancada(relogio, termica, qps=200),
                            "termica": termica},
        ao_progresso=lambda i, f, e, p: progresso.append((i, f, e)),
        ao_terminar=lambda i, laudo: laudos.__setitem__(i, laudo),
        intervalo_progresso_s=0.0)
    executor.iniciar("e1", "calor", {"duracao_min": 60, "janela_s": 0.1, "amostra_s": 0.1})
    with pytest.raises(RuntimeError, match="em curso"):
        executor.iniciar("e2", "rede", {})
    limite = time.monotonic() + 5
    while not any(e.startswith("janela") for _, _, e in progresso):
        assert time.monotonic() < limite
        time.sleep(0.02)
    assert executor.cancelar("e1") and not executor.cancelar("outro")
    assert executor.aguardar(5)
    assert laudos["e1"]["status"] == "cancelado" and laudos["e1"]["aprovado"] is None
    assert executor.ocupado is None
    assert {i for i, _, _ in progresso} == {"e1"}
    with pytest.raises(ValueError, match="desconhecidos"):
        executor.iniciar("e3", "calor", {"x": 1})


def test_todos_os_ensaios_tem_parametros_com_padrao_valido() -> None:
    for nome, classe in CATALOGO.items():
        obrigatorios = {p.nome for p in classe.parametros if p.obrigatorio}
        if not obrigatorios:
            assert classe.valida({}), nome


# -- FASE 0 inteira na bancada (python -m valtexa_rover.sim fase0) ------------------------------


def test_fase0_na_bancada_aprova_e_o_laudo_diz_que_e_simulado(tmp_path) -> None:
    from valtexa_rover.ensaios.fase0 import laudo_markdown, rodar_na_bancada
    from valtexa_rover.sim.__main__ import main

    laudos = rodar_na_bancada(tmp_path / "b")
    situacao = {x["rotulo"]: x["status"] for x in laudos}
    assert situacao.pop("bateria e energia") == "nao_avaliado"
    assert set(situacao.values()) == {"aprovado"}, situacao
    assert len([r for r in situacao if r.startswith("profundidade")]) == 3
    texto = laudo_markdown(laudos, titulo="FASE 0")
    assert "SIMULADO / não testado em hardware" in texto and "aguarda protocolo" in texto
    saida = tmp_path / "laudo.md"
    assert main(["fase0", "--saida", str(saida)]) == 0 and saida.exists()


def test_fase0_na_bancada_com_imu_trocada_reprova_so_a_imu(tmp_path) -> None:
    from valtexa_rover.ensaios.fase0 import rodar_na_bancada

    laudos = rodar_na_bancada(tmp_path / "b", falhas={"imu": {"escala_g": 1.05}})
    reprovados = [x["rotulo"] for x in laudos if x["status"] == "reprovado"]
    assert reprovados == ["IMU parada"]


def test_cli_roda_um_ensaio_sem_plataforma(tmp_path, capsys) -> None:
    import json

    from valtexa_rover.cli import main

    hal = "valtexa_rover.ensaios.simulado:_bancada_cli"
    codigo = main(["ensaio", "imu", "--dados", str(tmp_path), "--hal", hal,
                   "--param", "duracao_s=2", "--param", "roll_esperado_deg=0"])
    laudo = json.loads(capsys.readouterr().out)
    assert codigo == 0 and laudo["status"] == "aprovado" and laudo["params"]["duracao_s"] == 2
    assert main(["ensaio", "voar", "--dados", str(tmp_path), "--hal", hal]) == 1
