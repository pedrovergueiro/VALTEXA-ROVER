"""Malha com repetidoras (Parte 13, item 2) — rádio EMULADO, relógio virtual, sementes fixas.

Topologias com Rover, repetidoras e base; nós derrubados no meio da missão. Exigências:
zero perda de P0 a P3 (uma vez só), rota pela qualidade medida, troca de rota sozinha
quando um nó morre, guarda-e-encaminha, teleop só o último e nunca vencido.
"""

from __future__ import annotations

from dataclasses import replace

from valtexa_rover.enlace import rpc
from valtexa_rover.enlace.bancada import ALTA, SUBGHZ
from valtexa_rover.enlace.bancada_malha import BancadaMalha, Ligacao, Placar, resumo
from valtexa_rover.enlace.malha import MAX_SALTOS, custo_do_enlace
from valtexa_rover.enlace.no import P0, P1, P2, P3, P4
from valtexa_rover.enlace.saude import SaudeDoEnlace

BOM = replace(ALTA, banda_bps=5e6, atraso_s=0.005, variacao_s=0.003, perda=0.05)
RUIM = replace(SUBGHZ, banda_bps=250e3, perda=0.35)


def _linha(tmp_path, **kw) -> BancadaMalha:
    """Rover → r1 → r2 → base (o Rover não ouve a base)."""
    return BancadaMalha(tmp_path, {"rover": "rover", "r1": "repetidora", "r2": "repetidora",
                                   "base": "base"},
                        [Ligacao("rover", "r1", "alta", BOM), Ligacao("r1", "r2", "alta", BOM),
                         Ligacao("r2", "base", "alta", BOM)], niveis={"r1": 2, "r2": 1}, **kw)


def _losango(tmp_path, direto: bool = False, atraso_b: float = 0.005) -> BancadaMalha:
    """Rover ouve rA e rB; as duas ouvem a base. ``direto``: um sub-GHz ruim Rover-base."""
    lig = [Ligacao("rover", "rA", "alta", BOM),
           Ligacao("rover", "rB", "alta", replace(BOM, perda=0.12, atraso_s=atraso_b)),
           Ligacao("rA", "base", "alta", BOM), Ligacao("rB", "base", "alta", BOM)]
    if direto:
        lig.append(Ligacao("rover", "base", "subghz", RUIM))
    return BancadaMalha(tmp_path, {"rover": "rover", "rA": "repetidora", "rB": "repetidora",
                                   "base": "base"}, lig)


def _manda_lote(placar: Placar, n: int, tamanho: int = 600) -> None:
    for i in range(n):
        placar.manda("rover", "base", i % 4, tamanho + 37 * i)
        placar.manda("base", "rover", i % 4, tamanho)


def test_custo_do_enlace_cresce_com_perda_rtt_e_banda_baixa() -> None:
    bom = SaudeDoEnlace("a", 5e6, vivo=True, srtt=0.01, perda=0.0)
    perda = SaudeDoEnlace("a", 5e6, vivo=True, srtt=0.01, perda=0.5)
    lento = SaudeDoEnlace("a", 250e3, vivo=True, srtt=0.01, perda=0.0)
    longe = SaudeDoEnlace("a", 5e6, vivo=True, srtt=0.4, perda=0.0)
    morto = SaudeDoEnlace("a", 5e6, vivo=False)
    assert custo_do_enlace(bom) < min(custo_do_enlace(perda), custo_do_enlace(lento),
                                      custo_do_enlace(longe))
    assert abs(custo_do_enlace(perda) - 2 * custo_do_enlace(bom)) < 1e-9  # ETX: 1/(1-perda)
    assert custo_do_enlace(morto) is None


def test_tres_saltos_nada_se_perde_nos_dois_sentidos(tmp_path) -> None:
    b = _linha(tmp_path)
    b.anda(5)
    assert b.malha("rover").rotas["base"].saltos == 3
    assert b.malha("rover").rotas["base"].prox == "r1"
    placar = Placar(b)
    _manda_lote(placar, 120)
    b.ate(lambda: not placar.faltam(), 60, placar.colhe)
    b.anda(3, placar.colhe)
    assert placar.faltam() == [] and placar.repetidas() == []
    # a origem só apaga a cópia com a confirmação fim a fim
    assert b.malha("rover")._pendentes_para("base") == 0
    assert b.malha("base")._pendentes_para("rover") == 0
    assert b.malha("r1").estat["malha_encaminhadas"] > 0


def test_escolhe_a_rota_pela_qualidade_medida_e_nao_pelo_numero_de_saltos(tmp_path) -> None:
    b = _losango(tmp_path, direto=True)
    b.anda(8)
    r = b.malha("rover").rotas["base"]
    assert r.prox == "rA" and r.saltos == 2  # 2 saltos bons valem mais que 1 ruim
    assert b.malha("rover").segundas["base"].prox in ("rB", "base")


def test_repetidora_morre_no_meio_da_missao_a_rota_troca_sozinha_sem_perder_nada(
        tmp_path) -> None:
    b = _losango(tmp_path)
    b.anda(5)
    assert b.malha("rover").rotas["base"].prox == "rA"
    placar = Placar(b)
    prox_envio = [0.0]

    def missao() -> None:  # telemetria, resultados e comandos o tempo todo
        agora = b.relogio.agora()
        if agora >= prox_envio[0]:
            prox_envio[0] = agora + 0.2
            for prio in (P1, P2, P3):
                placar.manda("rover", "base", prio, 900)
            placar.manda("base", "rover", P1, 300)
        placar.colhe()

    b.anda(10, missao)
    t_queda = b.relogio.agora()
    b.mata("rA")  # bateria acabou no meio da missão, com mensagens dentro dela
    troca = b.ate(lambda: b.malha("rover").rotas.get("base") is not None
                  and b.malha("rover").rotas["base"].prox == "rB", 15, missao)
    b.anda(20, missao)
    b.ate(lambda: not placar.faltam(), 60, placar.colhe)
    assert placar.faltam() == [] and placar.repetidas() == []
    assert troca < 5.0, troca  # morto em ~3 s (sem batimento) + reavaliação
    # comandos (P1) que saíram depois da queda: atraso limitado pela redundância
    depois = [placar.recebidas[k][0] - t for k, (_, _, p, t) in placar.enviadas.items()
              if p == P1 and t > t_queda and k in placar.recebidas]
    assert resumo(depois)["p99_ms"] < 6000
    trocas = [t for t in b.malha("rover").trocas if t["dest"] == "base" and t["de"] == "rA"]
    assert trocas and trocas[0]["para"] == "rB"
    # a repetidora volta (com o que tinha guardado): nada chega duas vezes
    b.revive("rA")
    b.anda(30, missao)
    b.ate(lambda: not placar.faltam(), 60, placar.colhe)
    b.anda(5, placar.colhe)
    assert placar.faltam() == [] and placar.repetidas() == []


def test_repetidora_guarda_e_encaminha_quando_o_proximo_salto_volta(tmp_path) -> None:
    """Linha sem alternativa: r2 cai; r1 guarda (no disco) e entrega quando r2 voltar."""
    b = _linha(tmp_path)
    b.anda(5)
    b.mata("r2")
    placar = Placar(b)
    for i in range(30):
        placar.manda("rover", "base", P2 if i % 2 else P3, 2000)
    b.anda(10, placar.colhe)
    assert len(placar.faltam()) == 30  # sem caminho: nada chega, nada se perde
    b.mata("r1")
    b.revive("r1")  # e a repetidora ainda reinicia no meio (energia solar)
    b.revive("r2")
    b.ate(lambda: not placar.faltam(), 60, placar.colhe)
    b.anda(3, placar.colhe)
    assert placar.faltam() == [] and placar.repetidas() == []


def test_volta_para_a_rota_melhor_com_histerese_sem_oscilar(tmp_path) -> None:
    b = _losango(tmp_path, atraso_b=0.06)  # rB bem pior (RTT): vale voltar para rA
    b.anda(5)
    b.corta("rover", "rA", "alta", 10.0, 25.0)
    b.anda(40)
    trocas = [t for t in b.malha("rover").trocas if t["dest"] == "base" and t["de"]]
    assert [t["para"] for t in trocas][:2] == ["rB", "rA"]
    assert len(trocas) <= 3, trocas  # foi e voltou, sem ficar trocando


def test_diferenca_pequena_nao_troca_de_rota(tmp_path) -> None:
    """rB só um pouco pior (12 % contra 5 % de perda): depois de usar rB, não volta (histerese)."""
    b = _losango(tmp_path)
    b.anda(5)
    b.corta("rover", "rA", "alta", 10.0, 25.0)
    b.anda(40)
    trocas = [t for t in b.malha("rover").trocas if t["dest"] == "base" and t["de"]]
    assert [t["para"] for t in trocas] == ["rB"]


def test_repetidora_com_bateria_fraca_perde_o_trafego_antes_de_apagar(tmp_path) -> None:
    b = _losango(tmp_path)
    b.anda(5)
    assert b.malha("rover").rotas["base"].prox == "rA"
    b.nos["rA"].custo_extra_s = 1.0  # o controlador de carga avisou: bateria fraca
    b.anda(6)
    assert b.malha("rover").rotas["base"].prox == "rB"


def test_p0_p1_vao_pelas_duas_rotas_e_chegam_uma_vez(tmp_path) -> None:
    b = _losango(tmp_path)
    b.anda(5)
    placar = Placar(b)
    for _ in range(40):
        placar.manda("base", "rover", P0, 100)
        placar.manda("rover", "base", P1, 300)
    b.anda(5, placar.colhe)
    assert placar.faltam() == [] and placar.repetidas() == []
    assert b.malha("rover").estat["malha_repetidas"] > 0  # a outra cópia chegou e foi barrada
    assert b.malha("rA").estat["malha_encaminhadas"] > 0
    assert b.malha("rB").estat["malha_encaminhadas"] > 0


def test_teleop_pela_malha_so_o_ultimo_e_nunca_vencido(tmp_path) -> None:
    b = _linha(tmp_path)
    b.anda(5)
    aplicados: list[tuple[int, float]] = []
    for i in range(60):
        b.malha("base").enviar_ultimo("rover", "teleop", b"%d" % i, 1.0)
        b.anda(0.05)
        c = b.malha("rover").ultimo("base", "teleop")
        if c is not None:
            aplicados.append((int(c.corpo), b.relogio.agora()))
    assert len(aplicados) > 20
    assert [n for n, _ in aplicados] == sorted({n for n, _ in aplicados})  # nunca volta atrás
    b.corta("r1", "r2", "alta", b.relogio.agora(), b.relogio.agora() + 5)
    b.malha("base").enviar_ultimo("rover", "teleop", b"999", 1.0)
    b.anda(8)
    assert b.malha("rover").ultimo("base", "teleop") is None  # venceu no caminho


def test_rpc_sobre_a_malha_tres_saltos(tmp_path) -> None:
    """O mesmo RPC da P12 (pedido → base → resposta) com o "par" = destino final."""
    b = _linha(tmp_path)
    b.anda(5)
    respondidos = []

    def tratador(par, cab, corpo, prio):
        respondidos.append((par, cab["p"]))
        return 200, b"pong:" + corpo, {}

    cliente = rpc.ClienteRpc(b.malha("rover"), "base")
    msg = rpc.codifica({"rid": "x1", "m": "POST", "p": "/v1/device/sync", "h": {},
                        "prazo": 0}, b"ping")
    b.malha("rover").enviar("base", msg, P2, rpc.CANAL_PEDIDO, ttl_s=15)
    b.anda(3, lambda: rpc.atende(b.malha("base"), tratador))
    cliente._colhe(None)
    assert respondidos == [("rover", "/v1/device/sync")]
    status, corpo = cliente._respostas["x1"][0]["s"], cliente._respostas["x1"][1]
    assert status == 200 and corpo == b"pong:ping"
    assert b.malha("rover").vivo("base")


def test_sem_laco_infinito_quando_a_base_some(tmp_path) -> None:
    """Triângulo de repetidoras: a base morre; as mensagens não ficam rodando (reverso
    envenenado, teto de saltos) e chegam quando ela volta."""
    lig = [Ligacao("rover", "r1", "alta", BOM), Ligacao("r1", "r2", "alta", BOM),
           Ligacao("r2", "r3", "alta", BOM), Ligacao("r3", "r1", "alta", BOM),
           Ligacao("r2", "base", "alta", BOM), Ligacao("r3", "base", "alta", BOM)]
    b = BancadaMalha(tmp_path, {"rover": "rover", "r1": "repetidora", "r2": "repetidora",
                                "r3": "repetidora", "base": "base"}, lig,
                     niveis={"r1": 3, "r2": 1, "r3": 2})
    b.anda(6)
    placar = Placar(b)
    b.mata("base")
    for i in range(20):
        placar.manda("rover", "base", P3 if i % 2 else P1, 500)
    b.anda(20, placar.colhe)
    enc = sum(b.malha(r).estat["malha_encaminhadas"] for r in ("r1", "r2", "r3"))
    assert enc < 20 * MAX_SALTOS  # não ficou rodando
    assert all("base" not in b.malha(r).rotas for r in ("rover", "r1", "r2", "r3"))
    b.revive("base")
    b.ate(lambda: not placar.faltam((P3,)), 60, placar.colhe)
    assert placar.faltam((P3,)) == []  # P1 com validade pode ter vencido; P3 não se perde
    assert placar.repetidas() == []


def test_confirmacao_forjada_por_repetidora_nao_apaga_a_copia(tmp_path) -> None:
    b = _linha(tmp_path)
    b.anda(5)
    b.mata("r2")  # nada passa de r1
    placar = Placar(b)
    k = placar.manda("rover", "base", P3, 500)
    b.anda(2)
    # r1 (comprometida) inventa um "chegou" da base para o Rover
    from valtexa_rover.enlace.malha import CANAL_MALHA, _codifica

    e2e = b.malha("rover")._db.execute("SELECT id FROM malha_saida").fetchone()[0]
    falso = _codifica({"t": "a", "o": "base", "d": "rover", "s": 2, "h": "00" * 16}, e2e)
    b.malha("r1").no.enviar("rover", falso, P1, CANAL_MALHA, ttl_s=10)
    b.anda(2)
    assert b.malha("rover").estat.get("malha_ack_forjado", 0) == 1
    assert b.malha("rover")._pendentes_para("base") == 1  # a cópia continua
    b.revive("r2")
    b.ate(lambda: not placar.faltam(), 30, placar.colhe)
    assert k in placar.recebidas


def test_p4_pela_malha_so_com_a_sobra_e_retomado(tmp_path) -> None:
    b = _linha(tmp_path)
    b.anda(5)
    placar = Placar(b)
    for _ in range(6):
        placar.manda("rover", "base", P4, 150_000)
    for _ in range(40):
        placar.manda("rover", "base", P1, 200)
    b.anda(4, placar.colhe)
    assert not placar.faltam((P1,))
    b.mata("r1")
    b.anda(3)
    b.revive("r1")
    b.ate(lambda: not placar.faltam((P4,)), 120, placar.colhe)
    assert placar.faltam((P4,)) == [] and placar.repetidas() == []
    assert resumo(placar.latencias(P1))["p99_ms"] < 1500
    assert P0 == 0  # (nomes importados para a leitura)
