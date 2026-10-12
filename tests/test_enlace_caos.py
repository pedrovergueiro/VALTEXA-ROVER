"""Caos no enlace (rádio EMULADO, relógio virtual, sementes fixas).

Exigências da Parte 12: zero perda de P0 a P3; P4 retoma de onde parou; latência de P1
limitada sob carga (medida: p50/p95/p99); teleop vencido nunca aplicado; tudo volta
sozinho depois de quedas, troca de enlace e reinício dos processos.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import replace

from valtexa_rover.enlace.bancada import ALTA, SUBGHZ, Bancada, percentis
from valtexa_rover.enlace.emulador import ModeloDeCanal
from valtexa_rover.enlace.no import P1, P3, P4
from valtexa_rover.enlace.transferencia import EnviaArquivos, RecebeArquivos


class Placar:
    """Quem mandou o quê, quando; quem recebeu o quê, quando (e quantas vezes)."""

    def __init__(self, bancada: Bancada) -> None:
        self.b = bancada
        self.enviadas: dict[bytes, tuple[int, float]] = {}
        self.recebidas: dict[bytes, list[float]] = {}

    def manda(self, de: str, prio: int, tamanho: int, ttl_s: float | None = None) -> bytes:
        no, par = (self.b.rover, "base") if de == "rover" else (self.b.base, "rover")
        n = len(self.enviadas)
        cabeca = f"{de}:{prio}:{n:06d}|".encode()
        corpo = cabeca + os.urandom(max(0, tamanho - len(cabeca)))
        no.enviar(par, corpo, prio, ttl_s=ttl_s)
        self.enviadas[cabeca] = (prio, self.b.relogio.agora())
        return cabeca

    def colhe(self) -> None:
        agora = self.b.relogio.agora()
        for no in (self.b.rover, self.b.base):
            for c in no.chegadas(None, 1000):
                no.consumida(c.seq)
                cabeca = c.corpo[:c.corpo.index(b"|") + 1]
                self.recebidas.setdefault(cabeca, []).append(agora)

    def faltam(self, prios=(0, 1, 2, 3)) -> list[bytes]:
        return [k for k, (p, _) in self.enviadas.items() if p in prios and k not in
                self.recebidas]

    def repetidas(self) -> list[bytes]:
        return [k for k, v in self.recebidas.items() if len(v) > 1]

    def latencias(self, prio: int) -> list[float]:
        return [self.recebidas[k][0] - t for k, (p, t) in self.enviadas.items()
                if p == prio and k in self.recebidas]


def _com(modelo: ModeloDeCanal, **mudancas) -> ModeloDeCanal:
    return replace(modelo, **mudancas)


def _ate_entregar(b: Bancada, placar: Placar, limite_s: float, prios=(0, 1, 2, 3)) -> None:
    fim = b.relogio.agora() + limite_s
    while b.relogio.agora() < fim and placar.faltam(prios):
        b.anda(0.5, placar.colhe)


def test_30_por_cento_de_perda_nos_dois_enlaces_nada_se_perde(tmp_path) -> None:
    b = Bancada(tmp_path, {"subghz": _com(SUBGHZ, perda=0.3), "alta": _com(ALTA, perda=0.3)})
    placar = Placar(b)
    for i in range(400):
        placar.manda("rover" if i % 3 else "base", i % 4, 40 + (i * 397) % 20_000)
    _ate_entregar(b, placar, 120)
    assert placar.faltam() == []
    assert placar.repetidas() == []
    assert b.rover.estat["retransmissoes"] > 0  # a perda foi real
    assert b.rover.fila.tamanho_saida() == {} or b.rover.fila.tamanho_saida().get(0, 0) == 0


def test_rajadas_reordenacao_duplicacao_e_corrupcao(tmp_path) -> None:
    ruim = {"p_bom_ruim": 0.05, "p_ruim_bom": 0.2, "perda_ruim": 0.9, "reordena": 0.2,
            "reordena_s": 0.2, "duplica": 0.1, "corrompe": 0.05}
    b = Bancada(tmp_path, {"subghz": _com(SUBGHZ, **ruim), "alta": _com(ALTA, **ruim)},
                semente=11)
    placar = Placar(b)
    for i in range(300):
        placar.manda("rover" if i % 2 else "base", i % 4, 100 + (i * 911) % 30_000)
    _ate_entregar(b, placar, 180)
    assert placar.faltam() == [] and placar.repetidas() == []
    canais = [c for par in b.canais.values() for c in par]
    assert sum(c.est.corrompido for c in canais) > 0 and sum(c.est.duplicado for c in canais) > 0
    # corrupção é pega pelo CRC (antes da criptografia); repetição, pelo "vistos"
    assert b.base.estat["invalidos"] + b.rover.estat["invalidos"] > 0
    assert b.base.estat["repetidas_rx"] + b.rover.estat["repetidas_rx"] > 0


def test_quedas_de_5_a_60_s_e_troca_de_enlace_com_histerese(tmp_path) -> None:
    # alta cai de 10 s a 40 s (30 s) e de 70 s a 75 s; o sub-GHz é intermitente no fim
    b = Bancada(tmp_path, {"subghz": _com(SUBGHZ, perda=0.1, quedas=[(90.0, 150.0)]),
                           "alta": _com(ALTA, perda=0.1, quedas=[(10.0, 40.0), (70.0, 75.0)])})
    placar = Placar(b)
    proximo = [0.0]

    def carga() -> None:
        agora = b.relogio.agora()
        if agora >= proximo[0] and agora < 160:
            proximo[0] = agora + 0.5
            placar.manda("base", P1, 300)  # comando
            placar.manda("rover", P3, 3000)  # resultado
        placar.colhe()

    b.anda(170, carga)
    _ate_entregar(b, placar, 60)
    assert placar.faltam() == [] and placar.repetidas() == []
    trocas = b.rover.saude()["pares"]["base"]["trocas"]
    assert 2 <= trocas <= 6, trocas  # trocou nas quedas e voltou, sem ficar oscilando
    # P1 com só um enlace vivo continua com latência limitada (o outro estava caído)
    p1 = [placar.recebidas[k][0] - t for k, (p, t) in placar.enviadas.items()
          if p == P1 and 12 <= t < 38]
    assert p1 and percentis(p1)["p99_ms"] < 1500, percentis(p1)
    est = b.rover.saude()["pares"]["base"]["enlaces"]
    assert est["alta"]["quedas"] >= 2 and est["subghz"]["quedas"] >= 1


def test_queda_total_comando_vencido_e_recusado_e_tudo_volta_sozinho(tmp_path) -> None:
    b = Bancada(tmp_path, {"subghz": _com(SUBGHZ, quedas=[(5.0, 45.0)]),
                           "alta": _com(ALTA, quedas=[(5.0, 45.0)])})
    placar = Placar(b)
    b.anda(3, placar.colhe)
    assert b.rover.vivo("base")
    b.anda(5, placar.colhe)  # t = 8: as duas caídas
    assert not b.rover.vivo("base")
    assert b.rover.saude()["pares"]["base"]["sem_enlace_desde"] is not None
    vencido = placar.manda("base", P1, 200, ttl_s=10)  # vence no meio da queda
    guardados = [placar.manda("rover", p, 5000) for p in (0, 1, 2, 3) for _ in range(10)]
    b.anda(40, placar.colhe)  # t = 48: voltaram
    _ate_entregar(b, placar, 60)
    assert all(g in placar.recebidas for g in guardados)
    assert vencido not in placar.recebidas  # nunca aplicado
    assert b.base.estat["vencidas_tx"] + b.base.fila.descartes["expirada"] >= 1
    assert b.rover.vivo("base")


def test_matar_e_reiniciar_os_processos_no_meio_da_transferencia(tmp_path) -> None:
    b = Bancada(tmp_path, {"subghz": _com(SUBGHZ, perda=0.2), "alta": _com(ALTA, perda=0.2,
                                                                            banda_bps=2e6)})
    original = tmp_path / "original.jpg"
    original.write_bytes(os.urandom(3 * 1024 * 1024 + 1234))
    sha = hashlib.sha256(original.read_bytes()).hexdigest()
    placar = Placar(b)
    tx = EnviaArquivos(b.rover, "base", tmp_path / "tx.db", janela=8)
    rx = RecebeArquivos(b.base, tmp_path / "recebidos")
    tx.oferece(original, "orig-1")
    total = tx.progresso("orig-1")[1]
    progresso_ao_matar = []

    def volta() -> None:
        tx.bombeia()
        rx.processa()
        if int(b.relogio.agora() * 100) % 50 == 0:
            placar.manda("rover", P3, 2000)

    for matar in ("rover", "base", "os dois"):
        b.anda(4, volta)
        progresso_ao_matar.append(tx.progresso("orig-1")[0])
        if matar in ("rover", "os dois"):
            b.mata_rover()
            tx.no = b.rover
        if matar in ("base", "os dois"):
            b.mata_base()
            rx.no = b.base
    fim = b.relogio.agora() + 300
    while not rx.concluidos and b.relogio.agora() < fim:
        b.anda(1, volta)
    placar.colhe()
    _ate_entregar(b, placar, 60, prios=(3,))
    assert rx.concluidos and hashlib.sha256(rx.concluidos[0].read_bytes()).hexdigest() == sha
    # retomou de onde parou: cada pedaço foi enfileirado uma vez e escrito uma vez
    assert 0 < progresso_ao_matar[0] < total
    assert tx.pecas_enfileiradas == total and rx.pecas_escritas == total
    assert placar.faltam() == [] and placar.repetidas() == []
    assert b.rover.estat["apertos"] >= 1  # a base nova obrigou um aperto de mão novo


def test_latencia_de_p1_sob_carga_de_p4_com_30_por_cento_de_perda(tmp_path) -> None:
    """A medida registrada no PROGRESSO (rádio EMULADO: 250 kbit/s + 20 Mbit/s)."""
    b = Bancada(tmp_path, {"subghz": _com(SUBGHZ, perda=0.3), "alta": _com(ALTA, perda=0.3)})
    placar = Placar(b)
    proximo = [0.0]

    def carga() -> None:
        agora = b.relogio.agora()
        if agora >= proximo[0]:
            proximo[0] = agora + 0.25
            placar.manda("base", P1, 700)  # o ponto de teleop assinado tem ~700 B
            placar.manda("rover", P3, 3000)
        if b.rover.fila.tamanho_saida().get(P4, 0) < 8:
            placar.manda("rover", P4, 250_000)
        placar.colhe()

    b.anda(120, carga)
    _ate_entregar(b, placar, 30, prios=(1, 3))
    p1, p3 = percentis(placar.latencias(P1)), percentis(placar.latencias(P3))
    p4 = sum(1 for k, (p, _) in placar.enviadas.items() if p == P4 and k in placar.recebidas)
    print(f"\nP1 sob carga (30% de perda, EMULADO): {p1}\nP3: {p3}\nP4 entregues: {p4}")
    assert placar.faltam() == []
    assert p1["n"] >= 400 and p1["p99_ms"] < 1000 and p1["p50_ms"] < 150
    assert p4 > 50  # o lote andou com a sobra


def test_p4_so_usa_a_sobra_no_enlace_lento(tmp_path) -> None:
    b = Bancada(tmp_path, {"subghz": SUBGHZ})  # só o sub-GHz: 250 kbit/s
    placar = Placar(b)
    proximo = [0.0]

    def carga() -> None:
        agora = b.relogio.agora()
        if agora >= proximo[0]:
            proximo[0] = agora + 0.2
            placar.manda("rover", P3, 2500)  # ~100 kbit/s de resultados
        if b.rover.fila.tamanho_saida().get(P4, 0) < 4:
            placar.manda("rover", P4, 100_000)
        placar.colhe()

    b.anda(60, carga)
    _ate_entregar(b, placar, 20, prios=(3,))
    p3 = percentis(placar.latencias(P3))
    assert placar.faltam() == [] and p3["p95_ms"] < 500, p3
    p4 = sum(1 for k, (p, _) in placar.enviadas.items() if p == P4 and k in placar.recebidas)
    assert p4 >= 5  # ~150 kbit/s de sobra x 60 s


def test_teleop_so_o_ultimo_ponto_e_vencido_nunca_aplicado(tmp_path) -> None:
    b = Bancada(tmp_path, {"subghz": _com(SUBGHZ, perda=0.3, reordena=0.3, reordena_s=0.5,
                                          quedas=[(20.0, 26.0)]),
                           "alta": _com(ALTA, perda=0.3, reordena=0.3, reordena_s=0.5,
                                        quedas=[(20.0, 26.0)])})
    aplicados: list[tuple[int, float]] = []
    proximo = [0.0]
    seq = [0]

    def volta() -> None:
        agora = b.relogio.agora()
        if agora >= proximo[0] and agora < 40:
            proximo[0] = agora + 0.25
            seq[0] += 1
            b.base.enviar_ultimo("rover", "teleop", b"%d" % seq[0], ttl_s=1.0)
        c = b.rover.ultimo("base", "teleop")
        if c is not None:
            assert c.expira > b.relogio.parede()  # nunca vencido
            aplicados.append((int(c.corpo), agora))

    b.anda(45, volta)
    seqs = [s for s, _ in aplicados]
    assert seqs == sorted(set(seqs))  # só para a frente: nada velho, nada repetido
    assert len(seqs) > 50
    assert not [t for _, t in aplicados if 21.0 <= t < 26.0]  # na queda, nada
    depois = [s for s, t in aplicados if t >= 26.0]
    assert depois and min(depois) >= (26.0 - 1.0) / 0.25  # o que venceu na queda não volta
    # o ponto novo substitui o velho (nunca há dois do mesmo slot em voo) e não vai ao disco
    assert len(b.base._pares["rover"].slots_tx) <= 1
    assert b.base.fila.tamanho_saida() == {}
