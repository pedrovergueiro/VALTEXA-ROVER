"""Missão ciente do enlace (Parte 13, item 4) — perfil SIMULADO de cafezal, rádio EMULADO.

O Rover desce a rua x = 220 m (na sombra do morro a partir de y ≈ 90 m) a 0,8 m/s com a
qualidade de cada rádio tirada do perfil a cada segundo. Com o mapa de cobertura, ele
avisa a base antes de entrar, atravessa sem pausar (a política da missão deixa), registra
a travessia e sincroniza tudo na volta. Perda **inesperada** continua pausando (P12).
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from valtexa_rover.enlace import cafezal, rpc
from valtexa_rover.enlace.bancada import ALTA, SUBGHZ, Bancada
from valtexa_rover.enlace.levantamento import MapaDeCobertura
from valtexa_rover.enlace.no import P1, P2, P3
from valtexa_rover.enlace.politica import (
    ACAO_PAUSADO,
    ACAO_TRAVESSIA,
    ACAO_VOLTAR,
    PoliticaDaMissao,
    PoliticaDeEnlace,
)
from valtexa_rover.enlace.travessia import (
    EM_TRAVESSIA,
    LIVRE,
    MissaoCienteDoEnlace,
    rota_da_missao,
)
from valtexa_rover.hal.rover_v3 import NavegacaoPorArquivo

T = cafezal.Talhao()


def _missao(x: float, *ys: float, **extra) -> dict:
    pontos = [T.latlon(x, y) for y in ys]
    return {"mission_id": "m-cafe", "segments": [{"index": 0, "path": [[lo, la] for la, lo
                                                                     in pontos]}], **extra}


class Percurso:
    """Rover andando na rua, com o rádio do perfil, a política e a travessia ligadas."""

    def __init__(self, tmp_path, payload: dict, *, x: float = 220.0, y0: float = 160.0,
                 alvos: tuple[float, ...] = (0.0,), mapa: bool = True, v: float = 0.8) -> None:
        self.b = Bancada(tmp_path)
        self.x, self.y, self.v = x, y0, v
        self.alvos = list(alvos)
        self.payload = payload
        self.nav = NavegacaoPorArquivo(tmp_path / "run" / "navegacao.json",
                                       tmp_path / "run" / "parar.json")
        self.nav.set_mode("autonomo")
        self.avisos: list[dict] = []
        self.travessia = MissaoCienteDoEnlace(
            MapaDeCobertura(cafezal.mapa_simulado(T)) if mapa else None,
            registro=tmp_path / "travessias.jsonl", parede=self.b.rover.parede,
            antecedencia_s=20, avisar=self._avisa)
        self.pol = PoliticaDeEnlace(self.b.rover, "base", tmp_path / "run" / "enlace.json",
                                    controlador=self.nav, missao=lambda: self.payload,
                                    travessia=self.travessia, posicao=self.pos)
        self.acoes: list[tuple[float, str, str]] = []
        self.enviados: dict[bytes, int] = {}
        self.recebidos: set[bytes] = set()
        self.sincronizou_em: list[float] = []

    def _avisa(self, aviso: dict) -> None:
        self.avisos.append(aviso)
        self.b.rover.enviar("base", json.dumps(aviso).encode(), P1, rpc.CANAL_CONTROLE,
                            ttl_s=30)

    def pos(self) -> dict:
        lat, lon = T.latlon(self.x, self.y)
        return {"lat": lat, "lon": lon, "fix": "fixed", "v_ms": self.v,
                "rumo_deg": 180.0 if self.alvos and self.alvos[0] < self.y else 0.0}

    def _radio(self) -> None:
        for nome, base in (("subghz", SUBGHZ), ("alta", ALTA)):
            q = cafezal.qualidade(T, nome, self.x, self.y)
            for canal in self.b.canais[nome]:
                canal.modelo = replace(cafezal.modelo(base, q), quedas=[])

    def segundo(self, k: int) -> None:
        if self.alvos and self.y == self.alvos[0]:
            self.alvos.pop(0)
        if self.nav.mode() == "autonomo" and self.alvos:  # a navegação anda (o Runtime não)
            alvo = self.alvos[0]
            passo = min(self.v, abs(alvo - self.y))
            self.y += passo if alvo > self.y else -passo
        self._radio()
        for prio in (P2, P3):  # telemetria e resultados o tempo todo
            corpo = b"%d:%d" % (prio, k)
            self.b.rover.enviar("base", corpo, prio)
            self.enviados[corpo] = prio
        for _ in range(2):
            self.b.anda(0.5, self._colhe)
            filas = self.b.rover.fila.tamanho_saida()
            vazio = not any(n for p, n in filas.items() if p >= 2)
            estado = self.travessia.atualiza(self.pos(), rota=rota_da_missao(self.payload),
                                             v_ms=self.v, vivo=self.b.rover.vivo("base"),
                                             filas=filas, vazio=vazio)
            if self.travessia.pede_sincronizar():
                self.sincronizou_em.append(self.b.relogio.agora())
            r = self.pol.aplicar()
            self.acoes.append((self.b.relogio.agora(), r["acao"], estado))

    def _colhe(self) -> None:
        for c in self.b.base.chegadas(None, 500):
            self.b.base.consumida(c.seq)
            if c.canal == rpc.CANAL_CONTROLE:
                continue
            self.recebidos.add(c.corpo)

    def roda(self, segundos: int) -> None:
        for k in range(segundos):
            self.segundo(k)


def test_politica_da_missao_explicita_derivada_da_plataforma_e_padrao() -> None:
    assert PoliticaDaMissao.de_payload(None) == PoliticaDaMissao()
    p = PoliticaDaMissao.de_payload({"politica_enlace": {"continuar_sem_enlace": True,
                                                         "travessia_max_s": 45}})
    assert p.continuar_sem_enlace and p.travessia_max_s == 45 and p.origem == "politica_enlace"
    spray = PoliticaDaMissao.de_payload({"command": {"offline_policy": {
        "grace_seconds": 30, "on_link_loss": "pause", "stop_application": True,
        "after_grace": "return_to_base"}}})
    assert not spray.atravessar_zonas_mapeadas and spray.fechar_aplicacao
    assert spray.voltar_apos_s == 30 and not spray.continuar_sem_enlace
    varredura = PoliticaDaMissao.de_payload({"offline_policy": {
        "grace_seconds": 300, "on_link_loss": "continue", "after_grace": "continue"}})
    assert varredura.atravessar_zonas_mapeadas and varredura.travessia_max_s == 120
    assert not varredura.continuar_sem_enlace  # perda inesperada ainda pausa (P12)


def test_atravessa_a_zona_morta_mapeada_sem_pausar_e_sincroniza_na_volta(tmp_path) -> None:
    """Desce a rua até a cabeceira (na sombra) e sobe de volta: ~225 s sem rádio."""
    p = Percurso(tmp_path, _missao(220, 160, 0, 160, politica_enlace={"travessia_max_s": 400}),
                 alvos=(0.0, 160.0))
    p.roda(440)
    acoes = {a for _, a, _ in p.acoes}
    assert ACAO_TRAVESSIA in acoes and ACAO_PAUSADO not in acoes
    assert p.nav.mode() == "autonomo"
    assert p.avisos and p.avisos[0]["zona"] == 1
    assert 180 <= p.avisos[0]["duracao_prevista_s"] <= 300
    # o aviso saiu (e a fila subiu) bem antes de o rádio cair
    t_aviso = next(t for t, _, e in p.acoes if e != LIVRE)
    t_entrou = next(t for t, _, e in p.acoes if e == EM_TRAVESSIA)
    assert t_entrou - t_aviso >= 10
    assert p.travessia.feitas == 1
    linha = json.loads((tmp_path / "travessias.jsonl").read_text().splitlines()[0])
    assert linha["zona"] == 1 and 150 < linha["real"]["duracao_s"] < 330
    assert linha["sincronizado_s_depois_de_sair"] is not None
    assert len(p.sincronizou_em) >= 2  # antes de entrar e na volta
    falta = [c for c in p.enviados if c not in p.recebidos]
    assert falta == []  # nada de P2/P3 se perdeu na travessia


def test_pulverizacao_nao_atravessa_as_cegas_pausa_e_fecha_a_aplicacao(tmp_path) -> None:
    spray = _missao(220, 160, 0, offline_policy={
        "grace_seconds": 30, "on_link_loss": "pause", "stop_application": True,
        "after_grace": "return_to_base"})
    p = Percurso(tmp_path, spray)
    p.roda(160)
    acoes = [a for _, a, _ in p.acoes]
    assert ACAO_PAUSADO in acoes and ACAO_TRAVESSIA not in acoes
    assert p.nav.mode() == "pausado"
    estado = json.loads((tmp_path / "run" / "enlace.json").read_text())
    assert estado["fechar_aplicacao"] is True and estado["politica"]["origem"] == \
        "offline_policy"
    assert ACAO_VOLTAR in acoes  # depois da graça: pede o retorno ao último ponto com enlace
    assert estado["retorno"]["lat"] == pytest.approx(T.latlon(220, 90)[0], abs=2e-4)


def test_perda_inesperada_fora_do_mapa_continua_pausando(tmp_path) -> None:
    p = Percurso(tmp_path, _missao(220, 160, 0, 160), mapa=False, alvos=(0.0, 160.0))
    p.roda(160)
    acoes = [a for _, a, _ in p.acoes]
    assert ACAO_PAUSADO in acoes and ACAO_TRAVESSIA not in acoes


def test_travessia_que_passa_muito_do_previsto_vira_perda_inesperada(tmp_path) -> None:
    p = Percurso(tmp_path, _missao(220, 160, 0, 160, politica_enlace={"travessia_max_s": 30}),
                 alvos=(0.0, 160.0))
    p.roda(220)
    acoes = [a for _, a, _ in p.acoes]
    i = acoes.index(ACAO_TRAVESSIA)
    assert ACAO_PAUSADO in acoes[i:]  # atravessou até o teto da missão e pausou


def test_servico_do_rover_liga_o_mapa_a_politica_e_avisa_a_base(tmp_path) -> None:
    from valtexa_rover.enlace.cripto import ChaveDeEnlace
    from valtexa_rover.enlace.emulador import RedeEmulada
    from valtexa_rover.enlace.transporte import RelogioVirtual
    from valtexa_rover.hal.rede_base import ServicoDeEnlace

    (tmp_path / "cobertura.json").write_text(json.dumps(cafezal.mapa_simulado(T)))
    relogio = RelogioVirtual()
    ponto = RedeEmulada(relogio).ponto("rover:alta", "alta")
    cfg = {"modo": "base", "nome": "RT-0001", "sincrono": "NORMAL",
           "base_pub": ChaveDeEnlace.gerar().publica.hex(),
           "cobertura": str(tmp_path / "cobertura.json"),
           "estado": str(tmp_path / "run" / "enlace.json"),
           "enlaces": [{"nome": "alta", "destino": "base-alta", "banda_bps": 1e6}]}
    servico = ServicoDeEnlace(cfg, tmp_path, relogio=relogio, transportes=[ponto])
    assert servico.travessia is not None and servico.travessia.mapa.zonas
    nav = NavegacaoPorArquivo(tmp_path / "run" / "navegacao.json",
                              tmp_path / "run" / "parar.json")
    servico.liga(nav)
    assert servico.politica.travessia is servico.travessia
    servico.missao_atual = _missao(220, 160, 0)
    lat, lon = T.latlon(220, 100)
    servico.atualiza_posicao({"lat": lat, "lon": lon, "fix": "fixed", "v_ms": 0.8,
                              "rumo_deg": 180.0})
    servico._atualiza_travessia()
    # sem base do outro lado o rádio está morto: já conta como dentro da travessia
    assert servico.travessia.estado in ("antecipando", "em_travessia")
    assert servico.no.fila.tamanho_saida().get(1) == 1  # o aviso à base está na fila (P1)
    assert servico._info()["pos"][0] == pytest.approx(lat)


def test_mapa_que_vem_assinado_na_missao_troca_o_da_travessia(tmp_path) -> None:
    """Sem ``cobertura.json`` no robô: o mapa chega na missão (a plataforma guarda o último
    levantamento da base) e a política passa a conhecer as zonas mortas na hora."""
    from valtexa_rover.enlace.cripto import ChaveDeEnlace
    from valtexa_rover.enlace.emulador import RedeEmulada
    from valtexa_rover.enlace.transporte import RelogioVirtual
    from valtexa_rover.hal.rede_base import ServicoDeEnlace

    relogio = RelogioVirtual()
    ponto = RedeEmulada(relogio).ponto("rover:alta", "alta")
    cfg = {"modo": "base", "nome": "RT-0001", "sincrono": "NORMAL",
           "base_pub": ChaveDeEnlace.gerar().publica.hex(),
           "estado": str(tmp_path / "run" / "enlace.json"),
           "enlaces": [{"nome": "alta", "destino": "base-alta", "banda_bps": 1e6}]}
    servico = ServicoDeEnlace(cfg, tmp_path, relogio=relogio, transportes=[ponto])
    servico.liga(NavegacaoPorArquivo(tmp_path / "run" / "navegacao.json",
                                     tmp_path / "run" / "parar.json"))
    assert servico.travessia is None and servico.politica.travessia is None

    # como a plataforma manda (services/cobertura_talhao.para_missao)
    mapa = {k: v for k, v in cafezal.mapa_simulado(T).items()
            if k in ("versao", "origem", "celula_m", "celulas", "zonas_mortas")}
    mapa["gerado_em"] = "2026-10-10T12:00:00+00:00"
    assert servico.cobertura_da_missao(mapa) is True
    assert servico.travessia is not None and servico.travessia.mapa.zonas
    assert servico.politica.travessia is servico.travessia  # a política já usa
    assert servico.cobertura_da_missao(dict(mapa)) is False  # o mesmo: nada a fazer

    # levantamento novo: troca o mapa na mesma travessia (o registro continua)
    travessia = servico.travessia
    assert servico.cobertura_da_missao({**mapa, "gerado_em": "2026-10-11T08:00:00+00:00",
                                        "zonas_mortas": []}) is True
    assert servico.travessia is travessia and travessia.mapa.zonas == []

    # mapa malformado não derruba: fica o anterior
    assert servico.cobertura_da_missao({"gerado_em": "x", "celulas": {"1,2": "bom"}}) is False
    assert travessia.mapa.zonas == []

    # missão sem mapa (e nada local): a travessia fica sem mapa, a política no padrão
    assert servico.cobertura_da_missao(None) is True and travessia.mapa is None
    lat, lon = T.latlon(220, 100)
    servico.atualiza_posicao({"lat": lat, "lon": lon, "fix": "fixed"})
    servico._atualiza_travessia()  # sem mapa não quebra
    assert travessia.estado == LIVRE
