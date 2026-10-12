"""Ensaio de 24 h **SIMULADO** (tempo acelerado) do enlace da fazenda — Parte 13, item 5.

**Nada aqui é medida de campo.** Relógio virtual, rádios emulados, perfil de cafezal em morro
(:mod:`.cafezal`). Serve para comparar versões do protocolo e da malha sob um dia inteiro de
falhas, e para dar ao Pedro os números que o ensaio de campo tem de bater.

Topologia (malha): Rover ⇄ base por sub-GHz e alta vazão (qualidade tirada do perfil na
posição do Rover a cada segundo: **oscila ao longo da rua** com a copa e some na sombra do
morro), Rover ⇄ repetidora solar no topo do morro (alta vazão; cobre a sombra) e
repetidora ⇄ base. Dia (padrão):

* 00:00-06:00 e 18:00-24:00 o Rover fica na doca, perto da base; 06:00-18:00 percorre as
  ruas em zigue-zague a 0,8 m/s;
* **Starlink cai** 01:00-01:45 (noite) e 15:10-15:40 (chuva); a chuva (15:00-16:00) também
  soma 5 % de perda no rádio de alta vazão;
* **queda de energia da base** 13:00-13:04 (processo morto, volta do disco);
* **repetidora morre** 10:00-10:50 (bateria) com o Rover trabalhando.

Tráfego (escala reduzida para caber no tempo do ensaio, dita no relatório): base → Rover P0
(sonda a cada 10 s, validade 2 s) e P1 (comando a cada 1 s, validade 4 s); Rover → base P2
(telemetria a cada 5 s) e P3 (resultado a cada 1 s em operação); P4 (original de 200 KB a
cada 1 min em operação). P2/P3 vão **até a nuvem** como a sincronização do Runtime: o Rover
guarda cada registro até a base dizer que a plataforma aceitou (com a Starlink fora, a base
responde "sem internet" e o Rover guarda); a nuvem deduplica pelo id.

Relatório: disponibilidade por classe (fração entregue dentro do prazo da classe, no enlace
e até a nuvem), latência de P1 (p50/p95/p99; resolução = passo do relógio), interrupção de P1
nas trocas de enlace/rota, tempo de troca de rota, e a conferência de **zero perda de P0 a
P3** (P2/P3 aceitos pela nuvem uma vez cada; P0/P1 entregues ou vencidos pela validade —
nunca perdidos com caminho disponível).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from valtexa_rover.enlace import cafezal
from valtexa_rover.enlace.bancada import ALTA, SUBGHZ
from valtexa_rover.enlace.bancada_malha import BancadaMalha, Ligacao
from valtexa_rover.enlace.fila import FilaCheiaError
from valtexa_rover.enlace.no import P0, P1, P2, P3, P4, ParametrosDoNo

C_DADO, C_ACK, C_SONDA = 20, 21, 22
PRAZO_S = {P0: 2.0, P1: 2.0, P2: 15.0, P3: 60.0}


@dataclass
class Perfil:
    horas: float = 24.0
    operacao_h: tuple[float, float] = (6.0, 18.0)
    starlink_fora_h: list[tuple[float, float, str]] = field(default_factory=lambda: [
        (1.0, 1.75, "noite"), (15.1667, 15.6667, "chuva")])
    chuva_h: list[tuple[float, float]] = field(default_factory=lambda: [(15.0, 16.0)])
    base_sem_energia_h: list[tuple[float, float]] = field(default_factory=lambda: [
        (13.0, 13.0667)])
    repetidora_morta_h: list[tuple[float, float]] = field(default_factory=lambda: [
        (10.0, 10.8333)])
    p0_s: float = 10.0
    p1_s: float = 1.0
    p2_s: float = 5.0
    p3_s: float = 1.0
    p4_s: float = 60.0
    p4_bytes: int = 200_000
    passo_operacao_s: float = 0.05
    passo_noite_s: float = 0.2
    ruas_inicio: int = 0
    """Primeira rua do zigue-zague (o teste curto começa perto da sombra do morro)."""
    drenagem_h: float = 0.5
    """Depois do dia, tudo no ar por este tempo: o que estava guardado tem de chegar."""
    v_ms: float = 0.8
    semente: int = 13

    @classmethod
    def curto(cls) -> Perfil:
        """2 h com todas as falhas (para a suíte de testes)."""
        return cls(horas=2.0, operacao_h=(0.05, 1.95),
                   starlink_fora_h=[(0.30, 0.40, "noite"), (1.55, 1.65, "chuva")],
                   chuva_h=[(1.5, 1.7)], base_sem_energia_h=[(0.70, 0.7667)],
                   repetidora_morta_h=[(1.00, 1.25)], drenagem_h=0.25, ruas_inicio=43)

    @classmethod
    def minimo(cls) -> Perfil:
        """40 min com todas as falhas, começando na sombra (para a suíte rápida)."""
        return cls(horas=0.6667, operacao_h=(0.01, 0.66),
                   starlink_fora_h=[(0.05, 0.10, "noite"), (0.55, 0.60, "chuva")],
                   chuva_h=[(0.5, 0.62)], base_sem_energia_h=[(0.20, 0.2333)],
                   repetidora_morta_h=[(0.35, 0.45)], drenagem_h=0.2, ruas_inicio=50)


def _dentro(h: float, janelas) -> bool:
    return any(a <= h < b for a, b, *_ in janelas)


class Ensaio24h:
    def __init__(self, pasta: Path, perfil: Perfil | None = None,
                 talhao: cafezal.Talhao | None = None) -> None:
        self.p = perfil or Perfil()
        self.t = talhao or cafezal.Talhao(repetidora_xy=(150.0, 95.0))
        rep_xy = self.t.repetidora_xy
        self.radios = {"subghz": SUBGHZ, "alta": ALTA}
        bom = replace(ALTA, perda=0.02)
        self.b = BancadaMalha(
            pasta, {"RT-0001": "rover", "rep": "repetidora", "base": "base"},
            [Ligacao("RT-0001", "base", "subghz", SUBGHZ),
             Ligacao("RT-0001", "base", "alta", ALTA),
             Ligacao("RT-0001", "rep", "alta", ALTA),
             Ligacao("rep", "base", "alta", bom)],
            semente=self.p.semente, parametros=ParametrosDoNo())
        self.rep_xy = rep_xy
        self.bom = bom
        self.rover = "RT-0001"
        self.caminho = None
        self.x, self.y = 10.0, 100.0  # doca
        # placar
        self.sondas: dict[bytes, tuple[int, float, str]] = {}
        self.entregas: dict[bytes, float] = {}
        self.pendentes_app: dict[str, tuple[int, bytes, float, float]] = {}
        self.nuvem: dict[str, int] = {}
        self.criados: dict[str, int] = {}
        self.p4_enviados: set[bytes] = set()
        self.p4_recebidos: set[bytes] = set()
        self.eventos: list[dict] = []
        self.rotas: list[tuple[float, str | None]] = []
        self.morte_fisica: dict[tuple[str, str], float] = {}
        """(vizinho do Rover, rádio) -> quando o canal emulado morreu de fato."""
        self.trocas_rota: list[dict] = []
        self.trocas_enlace: list[dict] = []
        self._medindo_rota: tuple[str, float] | None = None
        self._medindo_enlace: tuple[str, float] | None = None
        self.caminho_ok: list[tuple[float, str]] = []
        """A cada segundo: o caminho Rover ⇄ base estava ``bom`` (rota e o enlace do próximo
        salto com perda ≤ 30 %), ``degradado`` (rota, mas perda maior) ou ``sem``?"""
        self._n = 0
        self._prox: dict[str, float] = {}
        self._estado = {"starlink": True, "base": True, "rep": True}

    # -- relógio do ensaio ----------------------------------------------------------

    @property
    def agora(self) -> float:
        return self.b.relogio.agora()

    def _h(self) -> float:
        return self.agora / 3600.0

    # -- mundo ----------------------------------------------------------------------

    def _move(self, h: float) -> None:
        a, b = self.p.operacao_h
        if a <= h < b:
            n_ruas = int(self.t.largura_m // self.t.rua_m)
            if self.caminho is None:
                self.caminho = iter(cafezal.caminho(self.t, self.p.v_ms, 1.0, range(
                    self.p.ruas_inicio, n_ruas)))
            try:
                _, self.x, self.y = next(self.caminho)
            except StopIteration:
                self.caminho = iter(cafezal.caminho(self.t, self.p.v_ms, 1.0))
        else:
            self.caminho = None
            self.x, self.y = 10.0, 100.0

    def _radios(self, h: float) -> None:
        chuva = 0.05 if _dentro(h, self.p.chuva_h) else 0.0
        for (a, b, radio), canais in self.b.canais.items():
            if {a, b} == {"rep", "base"}:
                modelo = replace(self.bom, perda=self.bom.perda + chuva)
            else:
                de = self.rep_xy if "rep" in (a, b) else None
                q = cafezal.qualidade(self.t, radio, self.x, self.y, de)
                modelo = cafezal.modelo(self.radios[radio], q)
                if radio == "alta" and q["vivo"]:
                    modelo = replace(modelo, perda=min(0.9, modelo.perda + chuva))
            for c in canais:
                c.modelo = replace(modelo, quedas=[])
            if self.rover in (a, b):
                viz = b if a == self.rover else a
                chave = (viz, radio)
                morto = modelo.perda >= 1.0 or not self.b.nos[viz].vivo
                if morto:
                    self.morte_fisica.setdefault(chave, self.agora)
                else:
                    self.morte_fisica.pop(chave, None)

    def _falhas(self, h: float) -> None:
        star = not _dentro(h, self.p.starlink_fora_h)
        if star != self._estado["starlink"]:
            self._estado["starlink"] = star
            self.eventos.append({"t": self.agora, "evento": "starlink " + (
                "volta" if star else "cai")})
        base = not _dentro(h, self.p.base_sem_energia_h)
        if base != self._estado["base"]:
            self._estado["base"] = base
            (self.b.revive if base else self.b.mata)("base")
            self.eventos.append({"t": self.agora, "evento": "base " + (
                "volta" if base else "sem energia")})
        rep = not _dentro(h, self.p.repetidora_morta_h)
        if rep != self._estado["rep"]:
            self._estado["rep"] = rep
            (self.b.revive if rep else self.b.mata)("rep")
            self.eventos.append({"t": self.agora, "evento": "repetidora " + (
                "volta" if rep else "morre")})
        self.b.nos["base"].atributos = {"nuvem": star}

    # -- tráfego --------------------------------------------------------------------

    def _vence(self, nome: str, periodo: float) -> bool:
        if self.agora >= self._prox.get(nome, 0.0):
            self._prox[nome] = self.agora + periodo
            return True
        return False

    def _sonda(self, de: str, para: str, prio: int, ttl: float) -> None:
        self._n += 1
        corpo = b"s%d:%d" % (prio, self._n)
        m = self.b.malha(de)
        with contextlib.suppress(FilaCheiaError):  # fila cheia: conta como não entregue
            m.enviar(para, corpo, prio, C_SONDA, ttl_s=ttl)
        self.sondas[corpo] = (prio, self.agora, de)

    def _registro(self, prio: int, tam: int) -> None:
        self._n += 1
        rid = f"r{self._n}"
        self.criados[rid] = prio
        self.pendentes_app[rid] = (prio, rid.encode() + b"|" + b"." * tam, self.agora, -1e9)

    def _trafego(self, h: float) -> None:
        operando = self.p.operacao_h[0] <= h < self.p.operacao_h[1]
        base_viva = self._estado["base"]
        if base_viva and self._vence("p0", self.p.p0_s):
            self._sonda("base", self.rover, P0, PRAZO_S[P0])
        if base_viva and operando and self._vence("p1", self.p.p1_s):
            self._sonda("base", self.rover, P1, 4.0)
        if self._vence("p2", self.p.p2_s):
            self._registro(P2, 600)
            self._sonda(self.rover, "base", P2, 60.0)
        if operando and self._vence("p3", self.p.p3_s):
            self._registro(P3, 3000)
            self._sonda(self.rover, "base", P3, 300.0)
        if operando and self._vence("p4", self.p.p4_s):
            self._n += 1
            corpo = b"o%d|" % self._n + b"\0" * self.p.p4_bytes
            with contextlib.suppress(FilaCheiaError):
                self.b.malha(self.rover).enviar("base", corpo, P4, C_DADO)
                self.p4_enviados.add(corpo[:12])
        self._sincroniza()

    def _mede_trocas(self) -> None:
        """Tempo de troca **medido do instante físico** em que o caminho em uso morreu (o
        canal emulado ficou mudo ou o vizinho caiu) até o tráfego ter outro caminho."""
        m = self.b.malha(self.rover)
        r = m.rotas.get("base")
        prox = None if r is None else r.prox
        if self._medindo_rota is None and prox is not None:
            radios = [k for k in self.morte_fisica if k[0] == prox]
            if radios and len(radios) == len(self.b.nos[self.rover].vizinhos[prox]):
                self._medindo_rota = (prox, max(self.morte_fisica[k] for k in radios))
        elif self._medindo_rota is not None:
            antigo, t0 = self._medindo_rota
            if prox != antigo:
                self.trocas_rota.append({"de": antigo, "para": prox, "h": round(t0 / 3600, 4),
                                         "s": round(self.agora - t0, 2)})
                self._medindo_rota = None
            elif not any(k[0] == antigo for k in self.morte_fisica):
                self._medindo_rota = None  # voltou antes de trocar
        par = m.no._pares.get("base")
        if par is None:
            return
        atual = par.escolha.atual
        if self._medindo_enlace is None and atual is not None and \
                ("base", atual) in self.morte_fisica:
            outro_vivo = any(("base", r_) not in self.morte_fisica for r_ in self.radios
                             if r_ != atual)
            if outro_vivo:
                self._medindo_enlace = (atual, self.morte_fisica[("base", atual)])
        elif self._medindo_enlace is not None:
            antigo, t0 = self._medindo_enlace
            if atual != antigo:
                self.trocas_enlace.append({"de": antigo, "para": atual,
                                           "s": round(self.agora - t0, 2)})
                self._medindo_enlace = None
            elif ("base", antigo) not in self.morte_fisica:
                self._medindo_enlace = None

    def _sincroniza(self) -> None:
        """A sincronização do Runtime: manda o que está pendente quando a base diz que tem
        nuvem; reenvia o que não teve aceite em 60 s."""
        m = self.b.malha(self.rover)
        if not (m.vivo("base") and m.info_do_par("base").get("nuvem")):
            return
        if not self._vence("sync", 1.0):
            return
        n = 0
        for rid, (prio, corpo, criado, enviado) in list(self.pendentes_app.items()):
            if self.agora - enviado < 60.0:
                continue
            try:
                m.enviar("base", corpo, prio, C_DADO)
            except FilaCheiaError:
                break
            self.pendentes_app[rid] = (prio, corpo, criado, self.agora)
            n += 1
            if n >= 200:
                break

    def _colhe(self) -> None:
        for nome in ("RT-0001", "base"):
            no = self.b.nos[nome]
            if not no.vivo:
                continue
            m = no.malha
            acks: list[str] = []
            for c in m.chegadas(None, 2000):
                m.consumida(c.seq)
                if c.canal == C_SONDA:
                    self.entregas.setdefault(c.corpo, self.agora)
                elif c.canal == C_DADO and nome == "base":
                    if c.corpo.startswith(b"o"):
                        self.p4_recebidos.add(c.corpo[:12])
                        continue
                    rid = c.corpo.split(b"|", 1)[0].decode()
                    if self._estado["starlink"]:
                        self.nuvem[rid] = self.nuvem.get(rid, 0) + 1
                        acks.append(rid)
                elif c.canal == C_ACK and nome == "RT-0001":
                    for rid in json.loads(c.corpo):
                        self.pendentes_app.pop(rid, None)
            if acks:
                with contextlib.suppress(FilaCheiaError):
                    m.enviar(self.rover, json.dumps(acks).encode(), P1, C_ACK, ttl_s=60.0)

    # -- laço -----------------------------------------------------------------------

    def roda(self, progresso=None) -> dict:
        fim = (self.p.horas + self.p.drenagem_h) * 3600.0
        proximo_seg = 0.0
        t0 = time.monotonic()
        while self.agora < fim:
            h = self._h()
            if h >= self.p.horas:  # drenagem: tudo no ar, Rover na doca, sem tráfego novo
                for k in ("starlink", "base", "rep"):
                    if not self._estado[k]:
                        self._falhas(-1.0)
                        break
            if self.agora >= proximo_seg:
                proximo_seg += 1.0
                if h < self.p.horas:
                    self._falhas(h)
                    self._move(h)
                else:
                    self._move(-1.0)
                self._radios(h)
                m = self.b.malha(self.rover)
                r = m.rotas.get("base")
                prox = None if r is None else r.prox
                if not self.rotas or self.rotas[-1][1] != prox:
                    self.rotas.append((self.agora, prox))
                vivos = self.b.nos["base"].vivo
                ok = "sem"
                if vivos and prox is not None and self.b.malha("base").vivo(self.rover):
                    perdas = [s_.perda for s_ in m.no._pares[prox].saude.values() if s_.vivo]
                    # o rádio do próximo salto está de pé de fato? (a rota só cai 3 s depois
                    # de o rádio morrer: nesse intervalo não há caminho, só a rota velha)
                    fisico = any((prox, r_) not in self.morte_fisica
                                 for r_ in self.b.nos[self.rover].vizinhos[prox])
                    ok = ("bom" if fisico and perdas and min(perdas) <= 0.3
                          else "degradado" if fisico else "sem")
                if not self.caminho_ok or self.caminho_ok[-1][1] != ok:
                    self.caminho_ok.append((self.agora, ok))
                if progresso and int(self.agora) % 3600 == 0:
                    progresso(self.agora / 3600, time.monotonic() - t0)
            if h < self.p.horas:
                self._trafego(h)
            else:
                self._sincroniza()
            for n in self.b.nos.values():
                if n.vivo:
                    n.malha.passo()
            self._colhe()
            self._mede_trocas()
            operando = self.p.operacao_h[0] <= h < self.p.operacao_h[1]
            self.b.relogio.avanca(self.p.passo_operacao_s if operando or h >= self.p.horas
                                  else self.p.passo_noite_s)
        return self.relatorio(time.monotonic() - t0)

    # -- relatório ------------------------------------------------------------------

    def relatorio(self, duracao_real_s: float = 0.0) -> dict:
        disp, lat_p1 = {}, []
        for prio in (P0, P1, P2, P3):
            sondas = [(k, t) for k, (p, t, _) in self.sondas.items() if p == prio]
            no_prazo = sum(1 for k, t in sondas if k in self.entregas
                           and self.entregas[k] - t <= PRAZO_S[prio])
            entregues = sum(1 for k, _ in sondas if k in self.entregas)
            disp[f"P{prio}"] = {"sondas": len(sondas), "no_prazo": no_prazo,
                                "entregues": entregues,
                                "disponibilidade": round(no_prazo / len(sondas), 4)
                                if sondas else None,
                                "prazo_s": PRAZO_S[prio]}
            if prio == P1:
                lat_p1 = [self.entregas[k] - t for k, t in sondas if k in self.entregas]
        perdidos = {p: [rid for rid, pp in self.criados.items() if pp == p
                        and rid not in self.nuvem] for p in (P2, P3)}
        duplicados = sum(1 for v in self.nuvem.values() if v > 1)
        rov = self.b.malha(self.rover)
        tr = [t["s"] for t in self.trocas_rota if t["para"] is not None]
        te = [t["s"] for t in self.trocas_enlace]
        return {
            "simulado": True,
            "aviso": "rádio EMULADO e perfil de cafezal SIMULADO: não é medida de campo",
            "perfil": dict(self.p.__dict__),
            "duracao_simulada_h": self.p.horas,
            "duracao_real_s": round(duracao_real_s, 1),
            "disponibilidade_enlace": disp,
            "latencia_p1": _perc(lat_p1),
            "nuvem": {"registros_criados": len(self.criados), "aceitos": len(self.nuvem),
                      "duplicados_absorvidos": duplicados,
                      "perdidos_p2": len(perdidos[P2]), "perdidos_p3": len(perdidos[P3]),
                      "pendentes_no_rover": len(self.pendentes_app)},
            "p4": {"enviados": len(self.p4_enviados), "recebidos": len(self.p4_recebidos)},
            "trocas_de_rota": rov.trocas_por_destino.get("base", 0),
            "trocas_de_enlace_com_a_base": rov.no._pares["base"].escolha.trocas,
            "tempo_de_troca_de_rota_s": {"n": len(tr), "p50": _q(tr, 0.5), "p95": _q(tr, 0.95),
                                         "max": max(tr) if tr else None,
                                         "sem_alternativa": sum(
                                             1 for t in self.trocas_rota if t["para"] is None)},
            "tempo_de_troca_de_enlace_s": {"n": len(te), "p50": _q(te, 0.5),
                                           "p95": _q(te, 0.95), "max": max(te) if te else None},
            "interrupcoes_p1_s": self._interrupcoes(),
            "p0_p1_nao_entregues": self.p01_nao_entregues(),
            "eventos": [{"h": round(e["t"] / 3600, 3), "evento": e["evento"]}
                        for e in self.eventos],
            "zero_perda_p0_p3": not perdidos[P2] and not perdidos[P3] and not
            self.pendentes_app and self._p01_sem_perda(),
        }

    def _interrupcoes(self) -> dict:
        """Episódios em que os comandos P1 (1/s) não chegaram no prazo, com a causa.

        Episódio = sequência de comandos enviados que não chegaram em 2 s; duração = do
        primeiro que falhou até o primeiro que voltou a chegar no prazo. Causa: base sem
        energia; repetidora morta com o Rover na sombra; senão **troca** (enlace oscilando,
        rota mudando) — é o "tempo de troca de enlace" que conta.
        """
        sondas = sorted((t, k) for k, (p, t, _) in self.sondas.items() if p == P1)
        episodios, inicio = [], None
        for t, k in sondas:
            ok = k in self.entregas and self.entregas[k] - t <= PRAZO_S[P1]
            if not ok and inicio is None:
                inicio = t
            elif ok and inicio is not None:
                episodios.append((inicio, t))
                inicio = None
        if inicio is not None and sondas:
            episodios.append((inicio, sondas[-1][0]))
        por_causa: dict[str, list[float]] = {}
        for a, b in episodios:
            causa = self._causa(a, b)
            por_causa.setdefault(causa, []).append(b - a)
        saida = {"episodios": len(episodios)}
        for causa, d in sorted(por_causa.items()):
            saida[causa] = {"n": len(d), "p50_s": _q(d, 0.5), "p95_s": _q(d, 0.95),
                            "max_s": round(max(d), 2), "total_s": round(sum(d), 1)}
        return saida

    def _causa(self, a: float, b: float) -> str:
        def janela(lista):
            return any(x * 3600 <= b and a <= y * 3600 + 5 for x, y, *_ in lista)

        if janela(self.p.base_sem_energia_h):
            return "base_sem_energia"
        if janela(self.p.repetidora_morta_h):
            return "repetidora_morta"
        return "troca"

    def _caminho_na_janela(self, de: float, ate: float) -> str:
        """O pior estado do caminho entre ``de`` e ``ate``: ``sem``, ``degradado`` ou ``bom``."""
        ordem = {"sem": 0, "degradado": 1, "bom": 2}
        estado, pior = "sem", "bom"
        for t, ok in self.caminho_ok:
            if t > ate:
                break
            if t <= de:
                estado = ok
            elif ordem[ok] < ordem[pior]:
                pior = ok
        return min(estado, pior, key=ordem.__getitem__)

    def p01_nao_entregues(self) -> dict:
        """P0/P1 que não chegaram dentro da validade, pela causa: **sem caminho** (rádio
        morto, repetidora ou base fora), **caminho degradado** (rota, mas perda > 30 %: a
        validade curta venceu nas retransmissões) ou **com caminho bom** (defeito: tem de ser
        zero). Comando vencido nunca é aplicado; quem pediu repete com assinatura nova."""
        sem, degradado, bom = 0, 0, []
        for k, (prio, t, _) in self.sondas.items():
            if prio > P1 or k in self.entregas:
                continue
            ttl = PRAZO_S[P0] if prio == P0 else 4.0
            estado = self._caminho_na_janela(t, t + ttl)
            if estado == "bom":
                bom.append(round(t / 3600, 3))
            elif estado == "degradado":
                degradado += 1
            else:
                sem += 1
        return {"vencidos_sem_caminho": sem, "vencidos_caminho_degradado": degradado,
                "perdidos_com_caminho_bom": len(bom), "horas_dos_perdidos": bom[:20]}

    def _p01_sem_perda(self) -> bool:
        return self.p01_nao_entregues()["perdidos_com_caminho_bom"] == 0


def _q(v: list[float], q: float) -> float | None:
    if not v:
        return None
    s = sorted(v)
    return round(s[min(len(s) - 1, round(q * (len(s) - 1)))], 2)


def _perc(v: list[float]) -> dict:
    if not v:
        return {"n": 0}
    s = sorted(v)

    def p(q: float) -> float:
        return round(s[min(len(s) - 1, round(q * (len(s) - 1)))] * 1000, 1)

    return {"n": len(s), "p50_ms": p(0.5), "p95_ms": p(0.95), "p99_ms": p(0.99),
            "max_ms": round(s[-1] * 1000, 1), "media_ms": round(statistics.fmean(s) * 1000, 1)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m valtexa_rover.enlace.ensaio24h",
                                 description="Ensaio de 24 h SIMULADO do enlace da fazenda.")
    ap.add_argument("--curto", action="store_true", help="2 h com todas as falhas")
    ap.add_argument("--minimo", action="store_true", help="40 min com todas as falhas")
    ap.add_argument("--saida", default=None, help="arquivo JSON do relatório")
    args = ap.parse_args(argv)
    perfil = Perfil.minimo() if args.minimo else Perfil.curto() if args.curto else Perfil()
    with tempfile.TemporaryDirectory() as d:
        rel = Ensaio24h(Path(d), perfil).roda(
            lambda h, s: print(f"  {h:5.1f} h simuladas em {s:6.0f} s", flush=True))
    texto = json.dumps(rel, indent=2, ensure_ascii=False)
    if args.saida:
        Path(args.saida).write_text(texto, encoding="utf-8")
    print(texto)
    return 0 if rel["zero_perda_p0_p3"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
