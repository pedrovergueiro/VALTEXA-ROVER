"""chrony com GNSS: leitura do ``chronyc -c`` e conferência (formato do chrony 4.x)."""

from __future__ import annotations

import subprocess

from valtexa_rover.enlace import relogio

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
