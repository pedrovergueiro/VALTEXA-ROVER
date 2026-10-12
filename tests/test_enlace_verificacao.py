"""Verificação hostil do protocolo de enlace da P12 (Parte 13, item 1).

Cada teste é um ataque ou uma falha que o revisor procurou: replay entre sessões, nonce
repetido depois de reiniciar, filas sem limite, disco cheio, relógio do Rover errado,
deduplicação depois de reiniciar, P4 matando P1 de fome, teleop vencido. Rádio EMULADO,
relógio virtual. Registro: ``docs/VERIFICACAO.md`` (seção "Parte 13").
"""

from __future__ import annotations

import os
import secrets
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from valtexa_rover.enlace import rpc
from valtexa_rover.enlace.bancada import ALTA, SUBGHZ, Bancada, percentis
from valtexa_rover.enlace.fila import DiscoCheioError, FilaCheiaError
from valtexa_rover.enlace.no import P0, P1, P3, P4
from valtexa_rover.enlace.quadro import DADOS, Cabecalho, abre_pacote, monta_pacote
from valtexa_rover.runtime.mission import (
    CommandRejectedError,
    PisoDaPlataforma,
    canonical_command,
    verify_command,
)
from valtexa_rover.runtime.store import LocalStore


def _grava_pacotes(b: Bancada, lado: str) -> list[tuple[str, bytes, str]]:
    """Copia tudo o que ``lado`` põe no ar: ``(enlace, pacote, destino)``."""
    gravados: list[tuple[str, bytes, str]] = []
    for nome in b.modelos:
        ponto = b.rede._pontos[f"{lado}:{nome}"]
        original = ponto.enviar

        def enviar(dados, destino, _orig=original, _nome=nome):
            gravados.append((_nome, dados, destino))
            _orig(dados, destino)

        ponto.enviar = enviar
    return gravados


def _colhe(no) -> list[bytes]:
    corpos = []
    for c in no.chegadas(None, 1000):
        no.consumida(c.seq)
        corpos.append(c.corpo)
    return corpos


# ---------------------------------------------------------------------------
# Replay e nonce
# ---------------------------------------------------------------------------


def test_replay_entre_sessoes_e_depois_de_reiniciar_nao_entrega_nada(tmp_path) -> None:
    b = Bancada(tmp_path)
    gravados = _grava_pacotes(b, "rover")
    for i in range(20):
        b.rover.enviar("base", b"cmd-%d" % i, P1)
    b.anda(3)
    assert len(_colhe(b.base)) == 20

    def reinjeta() -> list[bytes]:
        agora = b.relogio.agora()
        for enlace, pacote, _ in list(gravados):
            b.base._recebe(enlace, pacote, f"rover:{enlace}", agora)
        b.anda(0.5)
        return _colhe(b.base)

    # 1) mesma sessão: a janela anti-replay recusa
    assert reinjeta() == []
    # 2) sessão nova (troca de chave): a antiga sai da memória depois de duas trocas
    b.rover.p.rekey_s = 1.0
    b.anda(5)
    assert reinjeta() == []
    # 3) base reiniciada: não conhece mais a sessão — nem decifra
    b.mata_base()
    b.anda(3)
    assert reinjeta() == []
    assert b.base.estat["entregues_rx"] == 0  # a base nova não aceitou nada repetido


def test_nonce_nunca_se_repete_mesmo_reiniciando_os_dois_lados(tmp_path) -> None:
    b = Bancada(tmp_path)
    do_rover = _grava_pacotes(b, "rover")
    da_base = _grava_pacotes(b, "base")
    for rodada in range(3):
        for i in range(30):
            b.rover.enviar("base", b"r%d-%d" % (rodada, i), P3)
            b.base.enviar("rover", b"b%d-%d" % (rodada, i), P3)
        b.anda(4)
        b.mata_rover()
        b.mata_base()
    for gravados in (do_rover, da_base):
        vistos: set[tuple[bytes, int]] = set()
        for _, pacote, _ in gravados:
            cab, _, _ = abre_pacote(pacote)
            if cab.tipo != DADOS:
                continue
            chave = (cab.sessao, cab.contador)
            assert chave not in vistos, "nonce repetido na mesma sessão"
            vistos.add(chave)
        assert len({s for s, _ in vistos}) >= 3  # uma sessão (chaves novas) por reinício


def test_pacote_forjado_de_muitas_origens_nao_enche_a_memoria(tmp_path) -> None:
    b = Bancada(tmp_path)
    ponto = b.base.enlaces["alta"].transporte
    for i in range(10_000):
        lixo = monta_pacote(Cabecalho(DADOS, secrets.token_bytes(8), 1), os.urandom(40))
        b.base._recebe("alta", lixo, f"atacante:{i}", b.relogio.agora())
    assert len(b.base._ultimo_aviso) <= 4096
    assert ponto is not None


# ---------------------------------------------------------------------------
# Relógio do Rover errado
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("desvio_s", [-600.0, +600.0])
def test_relogio_do_rover_errado_nao_derruba_nem_estica_a_validade(tmp_path, desvio_s) -> None:
    """Antes: a base comparava o ``prazo`` (hora do Rover) com a hora dela — Rover 10 min
    atrasado = todo pedido "vencido" na base (comunicação morta); adiantado = pedido morto
    repassado. Agora a validade viaja como tempo restante e cada um usa o próprio relógio."""
    b = Bancada(tmp_path, desvio_parede_rover_s=desvio_s)
    cliente = rpc.ClienteRpc(b.rover, "base")
    respondidos: list[str] = []

    def tratador(par, cab, corpo, prio):
        respondidos.append(cab["p"])
        return 200, b"ok", {}

    def pede(caminho: str, prazo: float) -> str:
        rid = secrets.token_hex(4)
        msg = rpc.codifica({"rid": rid, "m": "GET", "p": caminho, "h": {},
                            "prazo": b.rover.relogio.parede() + prazo}, b"")
        b.rover.enviar("base", msg, P1, rpc.CANAL_PEDIDO, ttl_s=prazo)
        return rid

    rid = pede("/v1/device/commands", 4.0)
    b.anda(1.0, lambda: rpc.atende(b.base, tratador))
    cliente._colhe(None)
    assert respondidos == ["/v1/device/commands"]
    assert rid in cliente._respostas  # a resposta voltou dentro da validade
    # com o rádio caído mais que a validade, o pedido morre no caminho: não é repassado
    for c1, c2 in b.canais.values():
        c1.modelo.quedas.append((b.relogio.agora(), b.relogio.agora() + 6))
        c2.modelo.quedas.append((b.relogio.agora(), b.relogio.agora() + 6))
    pede("/v1/device/missions/current", 2.0)
    b.anda(12, lambda: rpc.atende(b.base, tratador))
    assert respondidos == ["/v1/device/commands"]


def test_passo_do_relogio_do_sistema_nao_estica_a_validade_na_fila(tmp_path) -> None:
    """O chrony acerta a hora para trás 1 h com um comando de 2 s na fila: ele ainda morre
    em 2 s (a validade do nó anda pelo relógio monotônico)."""
    b = Bancada(tmp_path)
    for c1, c2 in b.canais.values():
        c1.modelo.quedas.append((0, 60))
        c2.modelo.quedas.append((0, 60))
    b.rover.enviar("base", b"comando velho", P1, ttl_s=2.0)
    b.relogio._base -= 3600  # o relógio de parede "volta" 1 h
    b.anda(3)
    assert b.rover.fila.tamanho_saida() == {}  # venceu na fila, nunca sai
    b.anda(70)
    assert _colhe(b.base) == []


# ---------------------------------------------------------------------------
# Filas e disco cheio
# ---------------------------------------------------------------------------


def _enche_o_disco(fila) -> None:
    """``max_page_count`` = o tamanho atual e as páginas livres ocupadas: o SQLite passa a
    responder SQLITE_FULL, como num disco cheio de verdade."""
    db = fila._db
    db.execute("CREATE TABLE IF NOT EXISTS _lixo (x BLOB)")
    n = db.execute("PRAGMA page_count").fetchone()[0]
    db.execute(f"PRAGMA max_page_count = {n}")
    for tamanho in (65536, 4096, 512, 64):
        try:
            while True:
                db.execute("INSERT INTO _lixo VALUES (?)", (b"x" * tamanho,))
        except sqlite3.OperationalError:
            pass


def _libera_o_disco(fila) -> None:
    fila._db.execute("PRAGMA max_page_count = 1073741823")


def test_disco_cheio_no_receptor_parar_passa_e_o_resto_espera_sem_perder(tmp_path) -> None:
    b = Bancada(tmp_path)
    b.anda(2)
    _enche_o_disco(b.base.fila)
    b.rover.enviar("base", b"PARAR", P0, ttl_s=30)
    b.rover.enviar("base", b"r" * 20_000, P3)
    b.anda(3)
    chegou = _colhe(b.base)
    assert chegou == [b"PARAR"]  # P0 entregue da memória: o disco cheio não segura o PARAR
    assert b.base.estat["disco_cheio_rx"] >= 1
    assert b.rover.fila.tamanho_saida() == {3: 1}  # o P3 continua com o Rover, não se perdeu
    _libera_o_disco(b.base.fila)
    b.anda(10)
    assert _colhe(b.base) == [b"r" * 20_000]  # uma vez só
    b.anda(5)
    assert _colhe(b.base) == [] and b.rover.fila.tamanho_saida() == {}


def test_disco_cheio_no_emissor_p0_p1_vao_da_memoria_e_o_resto_e_recusado(tmp_path) -> None:
    b = Bancada(tmp_path)
    b.anda(2)
    _enche_o_disco(b.rover.fila)
    b.rover.enviar("base", b"comando" * 5000, P1, ttl_s=10)
    with pytest.raises(DiscoCheioError):
        b.rover.enviar("base", b"t" * 50_000, P3)
    assert issubclass(DiscoCheioError, FilaCheiaError)  # quem chamava já trata
    b.rover.enviar_ultimo("base", "teleop", b"ponto", 1.0)  # a sequência do slot não trava
    b.anda(2)
    assert b"comando" * 5000 in _colhe(b.base)
    assert b.rover.estat["disco_cheio_tx"] == 1


def test_entrada_vencida_que_ninguem_consumiu_e_limpa(tmp_path) -> None:
    b = Bancada(tmp_path)
    for i in range(50):
        b.rover.enviar("base", b"resposta-%d" % i, P1, rpc.CANAL_RESPOSTA, ttl_s=2)
    b.anda(3)
    assert b.base.fila.tamanho_entrada() == 50
    b.anda(700)  # a limpeza roda a cada 10 min
    assert b.base.fila.tamanho_entrada() == 0


# ---------------------------------------------------------------------------
# Prioridade e teleop
# ---------------------------------------------------------------------------


def test_p4_nao_mata_p1_de_fome_so_com_o_enlace_lento(tmp_path) -> None:
    """Só o sub-GHz (250 kbit/s) vivo, 10 % de perda, P4 sem parar: P1 a cada 250 ms."""
    b = Bancada(tmp_path, {"subghz": replace(SUBGHZ, perda=0.1)})
    enviados: dict[bytes, float] = {}
    lat: list[float] = []
    prox = [0.0]

    def carga() -> None:
        agora = b.relogio.agora()
        if agora >= prox[0]:
            prox[0] = agora + 0.25
            corpo = b"cmd-%09.3f" % agora
            enviados[corpo] = agora
            b.rover.enviar("base", corpo, P1)
        if b.rover.fila.tamanho_saida().get(4, 0) < 8:
            b.rover.enviar("base", b"o" * 100_000, P4)
        for c in b.base.chegadas(None, 500):
            b.base.consumida(c.seq)
            if c.prio == P1:
                lat.append(agora - enviados.pop(c.corpo))

    b.anda(60, carga)
    r = percentis(lat)
    assert enviados == {} or all(b.relogio.agora() - t < 1 for t in enviados.values())
    assert r["p99_ms"] < 700, r
    assert b.base.estat["entregues_rx"] > len(lat)  # o P4 também andou


def _assina(chave, payload: dict) -> dict:
    return {"payload": payload, "signature": chave.sign(canonical_command(payload)).hex()}


def _ponto(chave, seq: int, emitido: datetime, validade_s: float = 1.0) -> dict:
    return _assina(chave, {
        "algorithm": "x", "command": "device.teleop", "device_id": "RT-1", "sequence": seq,
        "nonce": secrets.token_hex(8), "issued_at": emitido.isoformat(),
        "expires_at": (emitido + timedelta(seconds=validade_s)).isoformat(),
        "body": {"v": 0.5, "w": 0.0,
                 "valid_until": (emitido + timedelta(seconds=validade_s)).isoformat()}})


class _Relogio:
    def __init__(self, agora: datetime) -> None:
        self.t, self.m = agora, 0.0

    def now(self) -> datetime:
        return self.t

    def monotonic(self) -> float:
        return self.m

    def anda(self, s: float) -> None:
        self.t += timedelta(seconds=s)
        self.m += s


def test_teleop_atrasado_pela_base_e_recusado_mesmo_com_o_relogio_do_rover_atrasado(
        tmp_path) -> None:
    chave = Ed25519PrivateKey.generate()
    pub = chave.public_key().public_bytes_raw().hex()
    t0 = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
    rover = _Relogio(t0 - timedelta(seconds=30))  # o relógio do Rover 30 s atrasado
    store = LocalStore(tmp_path / "s")
    piso = PisoDaPlataforma(rover.monotonic)
    kw = {"device_id": "RT-1", "platform_public_key": pub, "store": store,
          "sequence_key": "teleop", "command_prefix": "device.teleop"}
    guardado = _ponto(chave, 1, t0)  # a base "segura" este ponto
    verify_command(_ponto(chave, 2, t0 + timedelta(seconds=0.1)), now=rover.now(), piso=piso,
                   **kw)
    rover.anda(5)
    # Sem o piso, o ponto de 5 s atrás passaria (o relógio do Rover acha que são 11h59m35s)
    store2 = LocalStore(tmp_path / "s2")
    verify_command(_ponto(chave, 1, t0), now=rover.now(), **{**kw, "store": store2})
    # Com o piso (hora assinada da plataforma + tempo monotônico), não passa
    with pytest.raises(CommandRejectedError, match="vencido"):
        verify_command(_ponto(chave, 3, t0), now=rover.now(), piso=piso, **kw)
    assert piso.atraso_s(rover.now()) >= 29
    assert guardado["payload"]["sequence"] == 1
    # e um ponto novo continua passando
    verify_command(_ponto(chave, 4, t0 + timedelta(seconds=5)), now=rover.now(), piso=piso,
                   **kw)


def test_piso_ignora_envelope_sem_assinatura_valida(tmp_path) -> None:
    chave, outra = Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()
    pub = chave.public_key().public_bytes_raw().hex()
    rover = _Relogio(datetime(2026, 10, 9, 12, 0, tzinfo=UTC))
    piso = PisoDaPlataforma(rover.monotonic)
    falso = _ponto(outra, 9, rover.now() + timedelta(days=365))
    with pytest.raises(CommandRejectedError, match="assinatura"):
        verify_command(falso, device_id="RT-1", platform_public_key=pub,
                       store=LocalStore(tmp_path / "s"), now=rover.now(), piso=piso)
    assert piso.piso() is None  # um forjado não empurra a hora para frente (negar serviço)


def test_slot_velho_e_confirmado_para_o_emissor_parar_de_reenviar(tmp_path) -> None:
    b = Bancada(tmp_path, {"alta": replace(ALTA, reordena=0.5, reordena_s=0.3)})
    for i in range(40):
        b.rover.enviar_ultimo("base", "teleop", b"p%d" % i, 1.0)
        b.anda(0.05)
    b.anda(2)
    assert b.rover._pares["base"].slots_tx == {}
    assert not [m for m in b.rover._pares["base"].em_voo.values() if m.slot]


def test_enlace_calado_desvia_o_lote_em_1s_e_o_que_oscila_nao_puxa_de_volta() -> None:
    from valtexa_rover.enlace.saude import EscolhaDoMelhor, SaudeDoEnlace

    alta = SaudeDoEnlace("alta", 20e6, srtt=0.01)
    sub = SaudeDoEnlace("subghz", 250e3, srtt=0.05)
    esc = EscolhaDoMelhor()
    for t in range(5):
        alta.rx_valido(float(t))
        sub.rx_valido(float(t))
    assert esc.escolhe({"alta": alta, "subghz": sub}, 4.0) == "alta"
    sub.rx_valido(5.2)
    # alta calada há 1,2 s (ainda "viva" para P0/P1, morre só em 3 s): o lote já vai pelo sub
    assert esc.escolhe({"alta": alta, "subghz": sub}, 5.2) == "subghz"
    # alta oscila: morre e volta 3 vezes; ao voltar, espera ficar estável antes de puxar o lote
    t = 6.0
    for _ in range(3):
        alta.tick(t + 3.5)  # morreu
        t += 4.0
        for k in range(4):
            alta.rx_valido(t + k * 0.6)
            sub.rx_valido(t + k * 0.6)
        t += 2.0
        assert esc.escolhe({"alta": alta, "subghz": sub}, t) == "subghz"
    for k in range(60):  # estável por 30 s: volta para a alta
        alta.rx_valido(t + k * 0.5)
        sub.rx_valido(t + k * 0.5)
    assert esc.escolhe({"alta": alta, "subghz": sub}, t + 30.0) == "alta"
