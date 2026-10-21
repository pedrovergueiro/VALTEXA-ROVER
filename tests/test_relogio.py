"""chrony com GNSS: leitura do ``chronyc -c`` e conferência (formato do chrony 4.x)."""

from __future__ import annotations

import subprocess

from valtexa_rover.hal import relogio

TRACKING = ("50505300,PPS,1,1791000000.123,-0.000002500,0.000000400,0.000001100,"
            "-12.345,0.001,0.010,0.000000001,0.000010000,16.0,Normal\n")
SOURCES = ("#,*,PPS,0,4,377,10,0.000000010,0.000000012,0.000000100\n"
           "#,+,GPS,0,4,377,10,0.000120000,0.000121000,0.000200000\n"
           "^,-,a.ntp.br,2,6,0,-,0,0,0\n")


def test_le_e_avalia_chrony_com_pps() -> None:
    leitura = relogio.le_chrony(lambda a: TRACKING if a[-1] == "tracking" else SOURCES)
    assert leitura["tracking"]["ref_nome"] == "PPS" and leitura["tracking"]["stratum"] == 1
    assert [f["nome"] for f in leitura["fontes"]] == ["PPS", "GPS", "a.ntp.br"]
    r = relogio.avalia(leitura)
    assert r["sincronizado"] and r["fonte"] == "gnss" and r["fonte_em_uso"] == "PPS"
    assert r["offset_ms"] == -0.0025 and r["erro_estimado_ms"] < 0.02


def test_ntp_pela_internet_e_nao_sincronizado_e_sem_chrony() -> None:
    ntp = TRACKING.replace("PPS", "a.ntp.br").replace(",1,", ",3,", 1)
    r = relogio.avalia(relogio.le_chrony(
        lambda a: ntp if a[-1] == "tracking" else "^,*,a.ntp.br,2,6,377,10,0.0,0.0,0.01\n"))
    assert r["sincronizado"] and r["fonte"] == "ntp"
    solto = TRACKING.replace("Normal", "Not synchronised").replace(",1,", ",16,", 1)
    r = relogio.avalia(relogio.le_chrony(lambda a: solto if a[-1] == "tracking" else ""))
    assert not r["sincronizado"] and r["fonte"] == "nenhuma"

    def sem_chrony(args):
        raise FileNotFoundError("chronyc")

    r = relogio.avalia(relogio.le_chrony(sem_chrony))
    assert r["fonte"] == "nenhuma" and "chrony indisponível" in r["observacao"]

    def falha(args):
        raise subprocess.CalledProcessError(1, args)

    assert "erro" in relogio.le_chrony(falha)


# -- o ensaio ``relogio`` (FASE 0, só o Rover desde que o rádio da fazenda foi arquivado) ----


def _ensaio_relogio(tmp_path, le_chrony, params=None, core=None):
    from valtexa_rover.clock import SystemClock
    from valtexa_rover.ensaios import Contexto, rodar_por_nome
    from valtexa_rover.ensaios.relogio import FonteDoRelogio
    from valtexa_rover.ensaios.simulado import bancada

    clock = SystemClock()
    hardware, fontes = bancada(tmp_path / "bancada", clock, cameras=1)
    fontes["relogio"] = FonteDoRelogio(core, le_chrony=le_chrony)
    return rodar_por_nome("relogio", params, Contexto(hardware, clock, fontes))


def _chrony():
    """A leitura do chrony com a saída gravada (o ensaio chama sem argumentos)."""
    return relogio.le_chrony(lambda a: TRACKING if a[-1] == "tracking" else SOURCES)


def test_ensaio_relogio_aprova_o_rover_no_gnss(tmp_path) -> None:
    laudo = _ensaio_relogio(tmp_path, _chrony)
    assert laudo["status"] == "aprovado", laudo
    assert laudo["fase"] == 0
    nomes = {c["nome"] for c in laudo["criterios"]}
    assert nomes == {"rover_sincronizado", "rover_fonte_gnss", "rover_offset_ms"}  # sem base


def test_ensaio_relogio_reprova_o_offset_grande_e_o_atraso_da_plataforma(tmp_path) -> None:
    class Piso:
        def atraso_s(self, _agora) -> float:
            return 5.0

    class Core:
        piso_plataforma = Piso()
        clock = __import__("valtexa_rover.clock", fromlist=["SystemClock"]).SystemClock()

    laudo = _ensaio_relogio(tmp_path, _chrony, {"offset_max_ms": 0.001}, core=Core())
    ok = {c["nome"]: c["ok"] for c in laudo["criterios"]}
    assert laudo["status"] == "reprovado"
    assert ok["rover_offset_ms"] is False and ok["atraso_plataforma_s"] is False


def test_ensaio_relogio_sem_leitura_nao_se_aplica(tmp_path) -> None:
    from valtexa_rover.clock import SystemClock
    from valtexa_rover.ensaios import Contexto, rodar_por_nome
    from valtexa_rover.ensaios.simulado import bancada

    clock = SystemClock()
    hardware, fontes = bancada(tmp_path / "bancada", clock, cameras=1)
    fontes.pop("relogio", None)
    laudo = rodar_por_nome("relogio", None, Contexto(hardware, clock, fontes))
    assert laudo["status"] != "aprovado"


def test_relogio_na_fase0_da_bancada_e_com_defeito_injetado(tmp_path) -> None:
    from valtexa_rover.ensaios import Contexto, rodar_por_nome
    from valtexa_rover.ensaios.fase0 import etapas_fase0
    from valtexa_rover.ensaios.simulado import RelogioDeBancada, bancada

    assert "relogio" in {e.ensaio for e in etapas_fase0()}
    clock = RelogioDeBancada()
    for falhas, esperado in ((None, "aprovado"), ({"relogio": {"offset_s": 0.5}}, "reprovado"),
                             ({"relogio": {"fonte": "a.ntp.br"}}, "reprovado")):
        hardware, fontes = bancada(tmp_path / str(esperado) / str(bool(falhas)), clock,
                                   cameras=1, falhas=falhas)
        laudo = rodar_por_nome("relogio", None, Contexto(hardware, clock, fontes))
        assert laudo["status"] == esperado, (falhas, laudo.get("criterios"))
