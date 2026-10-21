"""O Rover "via base": RPC assinado sobre o enlace, política de perda de enlace e PARAR da base.

Rádio EMULADO (relé UDP no ``localhost`` com perda); nenhum rádio de verdade.
"""

from __future__ import annotations

import json
import socket
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from valtexa_rover._compat import UTC
from valtexa_rover.clock import ManualClock
from valtexa_rover.enlace import rpc
from valtexa_rover.enlace.bancada import ALTA, SUBGHZ, Bancada
from valtexa_rover.enlace.cripto import ChaveDeEnlace
from valtexa_rover.enlace.emulador import ModeloDeCanal, ReleUdp
from valtexa_rover.enlace.fila import FilaPersistente
from valtexa_rover.enlace.no import ConfigDoEnlace, NoDeEnlace
from valtexa_rover.enlace.politica import PoliticaDeEnlace
from valtexa_rover.enlace.transporte import Relogio, TransporteUdp
from valtexa_rover.hal.interfaces import NetworkUnavailableError
from valtexa_rover.hal.rede_base import ServicoDeEnlace
from valtexa_rover.hal.rover_v3 import NavegacaoPorArquivo
from valtexa_rover.runtime.client import DeviceClient, DeviceIdentity, canonical_request
from valtexa_rover.runtime.client import generate_keypair as par_ed25519


def _porta() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class BaseDeTeste:
    """Uma base mínima: nó respondedor + quem atende os pedidos (sem a nuvem)."""

    def __init__(self, pasta, chave, rovers: dict[str, bytes], portas: list[int],
                 tratador) -> None:
        self.relogio = Relogio()
        self.no = NoDeEnlace("base", chave, FilaPersistente(pasta / "base.db", self.relogio.parede,
                                                            sincrono="NORMAL"),
                             self.relogio,
                             [ConfigDoEnlace(f"e{i}", TransporteUdp(f"e{i}", ("127.0.0.1", p)),
                                             2e6) for i, p in enumerate(portas)],
                             rovers, iniciador=False, info_batimento=lambda: {"nuvem": True})
        self.ex = rpc.Executor(self.no, ao_passo=lambda: rpc.atende(self.no, tratador)).iniciar()

    def parar(self) -> None:
        self.ex.parar()
        self.no.fechar()


@pytest.fixture
def via_base(tmp_path):
    """Rover (ServicoDeEnlace de verdade, UDP) ⇄ 2 relés emulados com 20 % de perda ⇄ base."""
    k_base = ChaveDeEnlace.gerar()
    privada, publica = par_ed25519()
    recebidos: list[dict] = []
    modo = {"nuvem": True}

    def tratador(par, cab, corpo, prio):
        recebidos.append({"cab": cab, "corpo": corpo, "prio": prio, "par": par})
        if not modo["nuvem"]:
            return 503, b'{"detail":"sem internet"}', {rpc.CABECALHO_BASE: rpc.SEM_INTERNET}
        return 200, json.dumps({"eco": cab["p"], "tamanho": len(corpo)}).encode(), {
            "Content-Type": "application/json"}

    portas_base = [_porta(), _porta()]
    reles = [ReleUdp(("127.0.0.1", 0), ("127.0.0.1", p),
                     ModeloDeCanal(banda_bps=5e6, atraso_s=0.005, perda=0.2), semente=i
                     ).iniciar() for i, p in enumerate(portas_base)]
    cfg = {"modo": "base", "nome": "RT-0001", "base_pub": k_base.publica.hex(),
           "chave": str(tmp_path / "rover" / "enlace.key"), "fila": str(tmp_path / "rover.db"),
           "estado": str(tmp_path / "run" / "enlace.json"), "sincrono": "NORMAL",
           "enlaces": [{"nome": f"e{i}", "bind": f"127.0.0.1:{_porta()}",
                        "destino": f"{r.endereco[0]}:{r.endereco[1]}", "banda_bps": 5e6}
                       for i, r in enumerate(reles)]}
    servico = ServicoDeEnlace(cfg, tmp_path / "rover").iniciar()
    base = BaseDeTeste(tmp_path, k_base, {"RT-0001": servico.no.chave.publica}, portas_base,
                       tratador)
    identidade = DeviceIdentity("dev-1", "key-1", privada)
    relogio = ManualClock(datetime.now(UTC))
    cliente = DeviceClient(identidade, servico.rede, relogio)
    fim = time.monotonic() + 10
    while not servico.rede.is_online() and time.monotonic() < fim:
        time.sleep(0.05)
    yield {"servico": servico, "base": base, "cliente": cliente, "recebidos": recebidos,
           "modo": modo, "publica": publica, "reles": reles}
    servico.parar()
    base.parar()
    for r in reles:
        r.parar()


def test_requisicao_assinada_passa_pela_base_sem_mudar_um_byte(via_base) -> None:
    cliente = via_base["cliente"]
    assert via_base["servico"].rede.is_online()
    corpo = {"records": [{"id": "r1", "kind": "result", "body": {"x": "ç" * 3000}}]}
    r = cliente.request("POST", "/v1/device/edge/records", json_body=corpo)
    assert r.status == 200 and r.json()["eco"] == "/v1/device/edge/records"
    pedido = via_base["recebidos"][-1]
    assert pedido["prio"] == 3
    h = pedido["cab"]["h"]
    # a assinatura do Rover confere do outro lado: a base recebeu exatamente o que ele assinou
    texto = canonical_request("POST", "/v1/device/edge/records", "", h["X-Valtexa-Timestamp"],
                              h["X-Valtexa-Nonce"], pedido["corpo"])
    Ed25519PublicKey.from_public_bytes(bytes.fromhex(via_base["publica"])).verify(
        bytes.fromhex(h["X-Valtexa-Signature"]), texto)
    assert json.loads(pedido["corpo"]) == corpo
    # comandos em P1; teleop pelo slot "só o último vale"; original em P4
    assert cliente.request("GET", "/v1/device/commands").status == 200
    assert via_base["recebidos"][-1]["prio"] == 1
    assert cliente.request("GET", "/v1/device/teleop").status == 200
    assert via_base["recebidos"][-1]["cab"]["slot"] == "teleop"
    cliente.request("POST", "/v1/device/uploads", json_body={"upload_id": "u1",
                                                              "tier": "original"})
    cliente.request("PUT", "/v1/device/uploads/u1/chunks/0", raw=b"\xff" * 300_000,
                    content_type="application/octet-stream")
    assert via_base["recebidos"][-1]["prio"] == 4
    assert len(via_base["recebidos"][-1]["corpo"]) == 300_000


def test_base_sem_internet_vira_sem_rede_no_rover(via_base) -> None:
    via_base["modo"]["nuvem"] = False
    with pytest.raises(NetworkUnavailableError, match="sem internet"):
        via_base["cliente"].request("POST", "/v1/device/sync", json_body={"samples": []})
    assert not via_base["servico"].rede.is_online()  # a sincronização fica parada
    via_base["modo"]["nuvem"] = True
    time.sleep(1.1)
    assert via_base["servico"].rede.is_online()
    assert via_base["cliente"].request("POST", "/v1/device/sync", json_body={}).status == 200


def test_latencia_p1_pela_base_com_perda_real_em_udp(via_base) -> None:
    cliente = via_base["cliente"]
    tempos = []
    for _ in range(40):
        t = time.monotonic()
        assert cliente.request("GET", "/v1/device/commands").status == 200
        tempos.append(time.monotonic() - t)
    tempos.sort()
    print(f"\nRPC P1 via base (UDP, 2 relés emulados, 20% de perda): p50 "
          f"{tempos[20] * 1000:.1f} ms, p95 {tempos[37] * 1000:.1f} ms")
    assert tempos[37] < 2.0


def test_parar_difundido_pela_base_para_o_robo(via_base, tmp_path) -> None:
    nav = NavegacaoPorArquivo(tmp_path / "run" / "navegacao.json",
                              tmp_path / "run" / "parar.json")
    nav.set_mode("autonomo")
    servico = via_base["servico"]
    servico.liga(nav)
    base = via_base["base"].no
    base.enviar("RT-0001", json.dumps({"t": "parar", "motivo": "teste"}).encode(), 0,
                rpc.CANAL_CONTROLE, ttl_s=30)
    fim = time.monotonic() + 10
    while servico.paradas_da_base == 0 and time.monotonic() < fim:
        time.sleep(0.05)
    assert servico.paradas_da_base == 1
    assert nav.mode() == "parado"
    assert json.loads((tmp_path / "run" / "parar.json").read_text())["reason"] == "base: teste"
    assert json.loads(nav.teleop_path.read_text())["v"] == 0.0
    # a política escreve o enlace.json para a navegação
    fim = time.monotonic() + 5
    while not (tmp_path / "run" / "enlace.json").exists() and time.monotonic() < fim:
        time.sleep(0.05)
    estado = json.loads((tmp_path / "run" / "enlace.json").read_text())
    assert estado["algum_vivo"] is True and set(estado["enlaces"]) == {"e0", "e1"}


# ------------------------------------------------------------------ política (relógio virtual)


def _politica(tmp_path, quedas, payload):
    b = Bancada(tmp_path, {"subghz": replace(SUBGHZ, quedas=quedas),
                           "alta": replace(ALTA, quedas=quedas)})
    nav = NavegacaoPorArquivo(tmp_path / "run" / "navegacao.json",
                              tmp_path / "run" / "parar.json")
    nav.set_mode("autonomo")
    nav.teleop(0.3, 0.1, "2099-01-01T00:00:00+00:00", seq=9)
    pol = PoliticaDeEnlace(b.rover, "base", tmp_path / "run" / "enlace.json", controlador=nav,
                           missao=lambda: payload, tolerancia_s=2.0)
    return b, nav, pol


def _anda(b, pol, s):
    acoes = []
    b.anda(s, lambda: acoes.append(pol.aplicar()["acao"])
           if int(b.relogio.agora() * 100) % 25 == 0 else None)
    return acoes


def test_sem_enlace_pausa_a_missao_zera_teleop_e_volta_sozinho(tmp_path) -> None:
    b, nav, pol = _politica(tmp_path, [(5.0, 20.0)], {"mission_id": "m1"})
    assert set(_anda(b, pol, 4)) == {"nenhuma"}
    acoes = _anda(b, pol, 12)  # t = 16: sem enlace há ~11 s (morre ~3 s depois da queda)
    assert "pausado" in acoes and nav.mode() == "pausado"
    assert json.loads(nav.teleop_path.read_text())["v"] == 0.0
    estado = json.loads((tmp_path / "run" / "enlace.json").read_text())
    assert estado["algum_vivo"] is False and estado["sem_enlace_s"] > 2
    assert {e["estado"] for e in estado["enlaces"].values()} == {"morto"}
    acoes = _anda(b, pol, 10)  # religou
    assert "retomado" in acoes and nav.mode() == "autonomo"


def test_missao_que_permite_continua_autonoma_sem_enlace(tmp_path) -> None:
    b, nav, pol = _politica(tmp_path, [(5.0, 20.0)],
                            {"mission_id": "m2", "politica_enlace": {"continuar_sem_enlace": True}})
    acoes = _anda(b, pol, 16)
    assert "continua_autonomo" in acoes and nav.mode() == "autonomo"
    assert json.loads(nav.teleop_path.read_text())["v"] == 0.0  # teleop zera sempre


def test_queda_curta_dentro_da_tolerancia_nao_mexe_em_nada(tmp_path) -> None:
    b, nav, pol = _politica(tmp_path, [(5.0, 6.5)], {"mission_id": "m3"})
    assert set(_anda(b, pol, 12)) == {"nenhuma"} and nav.mode() == "autonomo"


def test_servico_monta_pelo_hardware_json_e_assina_a_chave(tmp_path) -> None:
    """``valtexa-rover enlace-chave`` e a montagem pelo bloco ``rede`` do ``hardware.json``."""
    import io
    from contextlib import redirect_stdout

    from valtexa_rover import cli
    from valtexa_rover.enlace.cripto import confere_publica_assinada

    privada, publica = par_ed25519()
    DeviceIdentity("RT-9", "k", privada).save(tmp_path / "identity.json")
    saida = io.StringIO()
    with redirect_stdout(saida):
        assert cli.main(["enlace-chave", "--identidade", str(tmp_path / "identity.json"),
                         "--chave", str(tmp_path / "enlace.key")]) == 0
    doc = json.loads(saida.getvalue())
    pub = confere_publica_assinada(doc, publica)
    exemplo = json.loads((Path(__file__).parents[1] / "deploy"
                          / "hardware.rover-v3.fazenda.json.exemplo").read_text(encoding="utf-8"))
    rede = exemplo["rede"]
    assert rede["modo"] == "base" and len(rede["enlaces"]) >= 2
    rede = {**rede, "base_pub": ChaveDeEnlace.gerar().publica.hex(),
            "chave": str(tmp_path / "enlace.key"), "fila": str(tmp_path / "e.db"),
            "enlaces": [{**e, "bind": f"127.0.0.1:{_porta()}", "destino": "127.0.0.1:9"}
                        for e in rede["enlaces"]]}
    s = ServicoDeEnlace(rede, tmp_path)
    try:
        assert s.no.chave.publica == pub  # a mesma chave que foi assinada
        assert list(s.no.enlaces) == [e["nome"] for e in rede["enlaces"]]
        assert s.rede.is_online() is False  # ninguém do outro lado
        with pytest.raises(NetworkUnavailableError):
            s.rede.send("GET", "/v1/device/commands", b"", {})
    finally:
        s.no.fechar()


def test_hal_do_rover_v3_com_rede_via_base(tmp_path) -> None:
    """``hardware.json`` com ``"rede": {"modo": "base"}`` → a HAL de rede é a ``RedeViaBase``."""
    from valtexa_rover.hal.rover_v3 import hal_rover_v3

    exemplo = json.loads((Path(__file__).parents[1] / "deploy"
                          / "hardware.rover-v3.fazenda.json.exemplo").read_text(encoding="utf-8"))
    exemplo["rede"].update({
        "base_pub": ChaveDeEnlace.gerar().publica.hex(), "chave": str(tmp_path / "e.key"),
        "fila": str(tmp_path / "e.db"), "estado": str(tmp_path / "run" / "enlace.json"),
        "enlaces": [{**e, "bind": f"127.0.0.1:{_porta()}", "destino": "127.0.0.1:9"}
                    for e in exemplo["rede"]["enlaces"]]})
    (tmp_path / "hardware.json").write_text(json.dumps(exemplo), encoding="utf-8")
    hw = hal_rover_v3(dados=tmp_path, url="https://nao-usado", clock=ManualClock(),
                      camera=lambda c: type("C", (), {"camera_id": c["id"]})(),
                      gnss=lambda g: g["porta"], imu=lambda i: i["canal"])
    try:
        assert isinstance(hw.network, rpc.RedeViaBase)
        assert list(hw.network.servico.no.enlaces) == ["subghz", "alta", "4g"]
        assert hw.network.describe().model == "via base (enlace de rádio)"
    finally:
        hw.network.servico.parar()
