"""Protocolo de enlace: quadros, criptografia, filas persistentes e limites.

Os testes de caos (perda, rajadas, quedas, troca de enlace, reinício no meio) estão em
``test_enlace_caos.py``.
"""

from __future__ import annotations

import os

import pytest

from valtexa_rover.enlace import quadro
from valtexa_rover.enlace.cripto import (
    ChaveDeEnlace,
    FalhaDeCriptoError,
    JanelaAntiReplay,
    conclui_aperto,
    confere_publica_assinada,
    id_do_par,
    inicia_aperto,
    publica_assinada,
    responde_aperto,
)
from valtexa_rover.enlace.fila import FilaCheiaError, FilaPersistente, Limites
from valtexa_rover.runtime.client import generate_keypair

# ---------------------------------------------------------------- quadros


def test_pacote_com_crc_recusa_qualquer_bit_trocado() -> None:
    cab = quadro.Cabecalho(quadro.DADOS, b"12345678", 42)
    pacote = quadro.monta_pacote(cab, b"corpo qualquer")
    lido, ad, corpo = quadro.abre_pacote(pacote)
    assert (lido.tipo, lido.sessao, lido.contador, corpo) == (quadro.DADOS, b"12345678", 42,
                                                             b"corpo qualquer")
    assert ad == cab.bytes()
    for i in range(len(pacote)):
        estragado = bytearray(pacote)
        estragado[i] ^= 0x40
        with pytest.raises(quadro.QuadroInvalidoError):
            quadro.abre_pacote(bytes(estragado))
    with pytest.raises(quadro.QuadroInvalidoError):
        quadro.abre_pacote(pacote[:10])


def test_registros_ida_e_volta() -> None:
    f = quadro.Frag(b"m" * 16, 1, 2, 3, 9, b"dados", ttl_ms=750, slot="teleop", slot_seq=77)
    a = quadro.Ack([(b"n" * 16, 5, 0b1011), (b"o" * 16, quadro.ACK_ENTREGUE, 0)])
    b = quadro.Bat(4, 10.5, 3, 9.5, 120, 0.002, b'{"nuvem":true}')
    corpo = (quadro.registro(quadro.R_FRAG, f.codifica())
             + quadro.registro(quadro.R_ACK, a.codifica())
             + quadro.registro(quadro.R_BAT, b.codifica()))
    lidos = quadro.le_registros(corpo)
    assert lidos[0] == f and lidos[1] == a and lidos[2] == b
    sem_ttl = quadro.Frag(b"m" * 16, 4, 0, 0, 1, b"x")
    assert quadro.le_registros(quadro.registro(quadro.R_FRAG, sem_ttl.codifica()))[0] == sem_ttl
    with pytest.raises(quadro.QuadroInvalidoError):  # índice >= total
        quadro.le_registros(quadro.registro(
            quadro.R_FRAG, quadro.Frag(b"m" * 16, 1, 0, 5, 2, b"").codifica()))


# ---------------------------------------------------------------- criptografia


def _aperto(ki: ChaveDeEnlace, kr: ChaveDeEnlace):
    pend = inicia_aperto(ki, kr.publica, 0.0)
    cab, ad, corpo = quadro.abre_pacote(pend.pacote)
    pub, s_r, hs2 = responde_aperto(kr, {id_do_par(ki.publica): ki.publica}, cab, ad, corpo,
                                    0.0)
    assert pub == ki.publica
    _, ad2, corpo2 = quadro.abre_pacote(hs2)
    s_i = conclui_aperto(ki, pend, ad2, corpo2, 0.0)
    return s_i, s_r


def test_aperto_de_mao_e_dados_cifrados_nos_dois_sentidos() -> None:
    ki, kr = ChaveDeEnlace.gerar(), ChaveDeEnlace.gerar()
    s_i, s_r = _aperto(ki, kr)
    p = s_i.sela(b"do rover")
    assert b"do rover" not in p
    cab, ad, corpo = quadro.abre_pacote(p)
    assert s_r.abre(cab, ad, corpo, 1.0) == b"do rover"
    cab, ad, corpo = quadro.abre_pacote(s_r.sela(b"da base"))
    assert s_i.abre(cab, ad, corpo, 1.0) == b"da base"


def test_replay_e_rejeitado_e_o_nonce_nunca_se_repete() -> None:
    ki, kr = ChaveDeEnlace.gerar(), ChaveDeEnlace.gerar()
    s_i, s_r = _aperto(ki, kr)
    pacotes = [s_i.sela(b"x%d" % n) for n in range(50)]
    contadores = [quadro.abre_pacote(p)[0].contador for p in pacotes]
    assert contadores == list(range(1, 51))  # nonce = contador: crescente, sem repetição
    # fora de ordem passa uma vez; a repetição não
    for p in reversed(pacotes):
        cab, ad, corpo = quadro.abre_pacote(p)
        s_r.abre(cab, ad, corpo, 0.0)
    for p in pacotes:
        cab, ad, corpo = quadro.abre_pacote(p)
        with pytest.raises(FalhaDeCriptoError):
            s_r.abre(cab, ad, corpo, 0.0)
    # sessão nova = chaves novas: o mesmo contador cifra diferente
    s_i2, _ = _aperto(ki, kr)
    assert s_i2.sela(b"x0")[20:-4] != pacotes[0][20:-4]


def test_janela_anti_replay_recusa_o_muito_velho() -> None:
    j = JanelaAntiReplay(64)
    j.marca(1000)
    assert not j.confere(1000) and not j.confere(900) and j.confere(990) and j.confere(1001)
    assert not j.confere(0)


def test_cabecalho_ou_corpo_adulterado_nao_decifra() -> None:
    ki, kr = ChaveDeEnlace.gerar(), ChaveDeEnlace.gerar()
    s_i, s_r = _aperto(ki, kr)
    cab, ad, corpo = quadro.abre_pacote(s_i.sela(b"PARAR"))
    with pytest.raises(FalhaDeCriptoError):
        s_r.abre(cab, ad, corpo[:-1] + bytes([corpo[-1] ^ 1]), 0.0)
    falso = quadro.Cabecalho(quadro.DADOS, cab.sessao, cab.contador + 1)
    with pytest.raises(FalhaDeCriptoError):
        s_r.abre(falso, falso.bytes(), corpo, 0.0)


def test_par_desconhecido_ou_chave_errada_nao_abre_sessao() -> None:
    ki, kr, intruso = ChaveDeEnlace.gerar(), ChaveDeEnlace.gerar(), ChaveDeEnlace.gerar()
    pend = inicia_aperto(intruso, kr.publica, 0.0)
    cab, ad, corpo = quadro.abre_pacote(pend.pacote)
    with pytest.raises(FalhaDeCriptoError):  # não está na lista de pares
        responde_aperto(kr, {id_do_par(ki.publica): ki.publica}, cab, ad, corpo, 0.0)
    # o intruso finge ser o par (mesmo id), sem a chave privada dele
    corpo_falso = id_do_par(ki.publica) + corpo[8:]
    with pytest.raises(FalhaDeCriptoError):
        responde_aperto(kr, {id_do_par(ki.publica): ki.publica}, cab, ad, corpo_falso, 0.0)


def test_chave_de_enlace_assinada_pela_identidade(tmp_path) -> None:
    privada, publica = generate_keypair()
    k = ChaveDeEnlace.carregar_ou_criar(tmp_path / "enlace.key")
    if os.name != "nt":  # no Windows o chmod não restringe; o Rover roda em Linux
        assert (tmp_path / "enlace.key").stat().st_mode & 0o777 == 0o600
    assert ChaveDeEnlace.carregar_ou_criar(tmp_path / "enlace.key").publica == k.publica
    assert "publica" in repr(k) and k._privada.private_bytes_raw().hex() not in repr(k)
    doc = publica_assinada(k, "RT-0001", privada)
    assert confere_publica_assinada(doc, publica) == k.publica
    outra = ChaveDeEnlace.gerar()
    with pytest.raises(FalhaDeCriptoError):
        confere_publica_assinada({**doc, "enlace_pub": outra.publica.hex()}, publica)
    _, outra_identidade = generate_keypair()
    with pytest.raises(FalhaDeCriptoError):
        confere_publica_assinada(doc, outra_identidade)


# ---------------------------------------------------------------- filas


class _Parede:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_fila_sobrevive_a_reabrir_e_deduplica(tmp_path) -> None:
    parede = _Parede()
    f = FilaPersistente(tmp_path / "f.db", parede)
    f.enfileira(b"a" * 16, "base", 2, 0, b"telemetria", None)
    f.enfileira(b"b" * 16, "base", 0, 0, b"batimento", None)
    assert f.recebe("base", b"c" * 16, 1, 1, b"comando", None) is True
    assert f.recebe("base", b"c" * 16, 1, 1, b"comando", None) is False  # repetida
    del f  # "kill": sem fechar
    g = FilaPersistente(tmp_path / "f.db", parede)
    assert [s.id for s in g.pendentes()] == [b"b" * 16, b"a" * 16]  # prioridade
    assert [c.corpo for c in g.chegadas()] == [b"comando"]
    assert g.ja_visto("base", b"c" * 16)
    assert g.recebe("base", b"c" * 16, 1, 1, b"comando", None) is False


def test_validade_vencida_nunca_sai_da_fila(tmp_path) -> None:
    parede = _Parede()
    f = FilaPersistente(tmp_path / "f.db", parede)
    f.enfileira(b"a" * 16, "base", 1, 0, b"comando", parede.t + 2)
    f.enfileira(b"b" * 16, "base", 1, 0, b"comando sem prazo", None)
    parede.t += 3
    assert [s.id for s in f.pendentes()] == [b"b" * 16]
    assert f.descartes["expirada"] == 1


def test_limites_p4_nunca_derruba_p0_a_p3(tmp_path) -> None:
    lim = Limites(limite_bytes=10_000, limite_p4_bytes=6_000, reserva_p01_bytes=2_000,
                  minimo_livre_bytes=0)
    f = FilaPersistente(tmp_path / "f.db", _Parede(), limites=lim)
    for i in range(6):
        f.enfileira(bytes([i]) * 16, "base", 4, 0, b"o" * 1000, None)
    with pytest.raises(FilaCheiaError):  # teto de P4
        f.enfileira(b"\xff" * 16, "base", 4, 0, b"o" * 1000, None)
    # P2/P3 entram tirando P4 da fila do enlace (o original continua na origem)
    for i in range(5):
        f.enfileira(bytes([100 + i]) * 16, "base", 3, 0, b"r" * 1500, None)
    assert f.tamanho_saida() == {3: 5, 4: 2}  # 6000 + 7500 > 10000: saíram 4 P4
    assert f.descartes["p4_removida"] == 4
    with pytest.raises(FilaCheiaError):  # tirar todo o P4 não bastaria: recusa, e nada cai
        f.enfileira(b"\xfe" * 16, "base", 2, 0, b"t" * 3000, None)
    assert f.tamanho_saida() == {3: 5, 4: 2}
    # P0/P1 têm reserva própria
    f.enfileira(b"\xfd" * 16, "base", 0, 0, b"p" * 1500, None)
    with pytest.raises(FilaCheiaError):
        f.enfileira(b"\xfc" * 16, "base", 1, 0, b"p" * 1000, None)


def test_disco_quase_cheio_so_entra_p0_p1_e_p4_sai(tmp_path) -> None:
    livre = [10**9]
    lim = Limites(minimo_livre_bytes=10**6)
    f = FilaPersistente(tmp_path / "f.db", _Parede(), limites=lim,
                        livre_no_disco=lambda: livre[0])
    f.enfileira(b"a" * 16, "base", 4, 0, b"o" * 100, None)
    f.enfileira(b"b" * 16, "base", 3, 0, b"r" * 100, None)
    livre[0] = 10
    with pytest.raises(FilaCheiaError):
        f.enfileira(b"c" * 16, "base", 3, 0, b"r", None)
    f.enfileira(b"d" * 16, "base", 0, 0, b"PARAR", None)
    assert f.tamanho_saida() == {0: 1, 3: 1}  # o P3 já aceito ficou; o P4 saiu


def test_contador_persistente(tmp_path) -> None:
    f = FilaPersistente(tmp_path / "f.db", _Parede())
    assert f.contador("slot:teleop", 1000) == 1000
    assert FilaPersistente(tmp_path / "f.db", _Parede()).contador("slot:teleop", 1000) == 2000
