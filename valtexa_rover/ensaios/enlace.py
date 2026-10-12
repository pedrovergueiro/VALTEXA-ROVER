"""Ensaios da **FASE ENLACE** (Parte 13): o rádio, a base, a Starlink e o relógio, medidos.

Todos usam a fonte ``enlace`` (:class:`FonteDoEnlace`, o serviço de enlace do Rover "via
base"); sem ela (rede HTTPS direta), o ensaio diz que não se aplica. Nenhum move o robô:
a travessia de zona morta é avaliada a partir das travessias que as missões registraram.

* ``enlace_rssi`` — RSSI/SNR (quando o rádio expõe), RTT e perda de cada rádio com a base,
  num ponto com posição RTK;
* ``enlace_latencia`` — ida e volta de P1 Rover → base → Rover pelo rádio (sem a nuvem):
  p50/p95/p99 e perda;
* ``enlace_troca`` — um rádio "morre" (a base o silencia por software, ou o técnico o
  desliga): tempo até o lote ir pelo outro e maior buraco de P1;
* ``enlace_starlink`` — a Starlink "cai" (por software na base, ou o cabo): nada se perde
  e o atraso acumulado entra em quanto tempo depois que ela volta;
* ``enlace_energia_base`` — o serviço da base cai e volta (ou o técnico corta a energia):
  em quanto tempo o enlace volta e nada se perde;
* ``enlace_travessia`` — as últimas travessias de zona morta mapeada: aviso com
  antecedência, duração dentro do previsto, sincronização na volta;
* ``relogio`` — chrony com GNSS no Rover e na base (sincronizado, fonte, offset) e o atraso
  contra a hora assinada da plataforma.

Pedidos que mexem na base (silenciar, Starlink, reiniciar) só valem com ``ensaios_remotos``
ligado no ``base.json`` — fora da validação fica desligado. **Simulado / não testado em
hardware**: só contra o emulador e UDP no ``localhost``; RSSI/SNR dependem do rádio
escolhido expor a leitura.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from valtexa_rover.enlace import relogio
from valtexa_rover.ensaios.base import (
    Contexto,
    Ensaio,
    EnsaioIndisponivelError,
    Param,
    criterio,
)
from valtexa_rover.ensaios.estatistica import mediana, percentil

FASE_ENLACE = 6
HW = ("simulado/não testado em hardware: rádios só no emulador; os números de campo vêm "
      "do levantamento e destes ensaios rodados no cafezal.")


class FonteDoEnlace:
    """O que os ensaios de enlace leem do serviço de enlace do Rover (e do Runtime)."""

    def __init__(self, servico, core=None, le_chrony=None) -> None:
        self.servico = servico
        self.core = core
        self.le_chrony = le_chrony or relogio.le_chrony

    def agora(self) -> float:
        return self.servico.relogio.agora()

    def vizinhos(self) -> dict:
        return self.servico.no.saude()["pares"]

    def posicao(self) -> dict | None:
        return self.servico.posicao

    def eco(self, timeout_s: float) -> float | None:
        t0 = self.agora()
        r = self.servico.pergunta_base({"t": "eco"}, timeout_s)
        return None if r is None else (self.agora() - t0) * 1000

    def pede(self, pedido: dict, timeout_s: float = 3.0) -> dict | None:
        return self.servico.pergunta_base(pedido, timeout_s)

    def melhor(self) -> str | None:
        par = self.servico.no._pares.get("base")
        return None if par is None else par.escolha.atual

    def ultimo_rx(self, enlace: str) -> float | None:
        par = self.servico.no._pares.get("base")
        s = None if par is None else par.saude.get(enlace)
        return None if s is None else s.ultimo_rx

    def nuvem_ok(self) -> bool:
        return bool(self.servico.rede.nuvem_ok())

    def travessias(self, n: int) -> list[dict]:
        t = self.servico.travessia
        if t is None or not Path(t.registro).exists():
            return []
        linhas = Path(t.registro).read_text(encoding="utf-8").splitlines()[-n:]
        return [json.loads(x) for x in linhas if x.strip()]

    def chrony(self) -> dict:
        return relogio.avalia(self.le_chrony())

    def atraso_plataforma_s(self) -> float | None:
        if self.core is None or not hasattr(self.core, "piso_plataforma"):
            return None
        return self.core.piso_plataforma.atraso_s(self.core.clock.now())


def _fonte(ctx: Contexto) -> FonteDoEnlace:
    f = ctx.fontes.get("enlace")
    if f is None:
        raise EnsaioIndisponivelError(
            "a rede deste robô não é 'via base' (rádio da fazenda): ensaio de enlace não se "
            "aplica")
    return f


def _pede_ou_falha(f: FonteDoEnlace, pedido: dict) -> dict:
    r = f.pede(pedido, 5.0)
    if r is None:
        raise EnsaioIndisponivelError(f"a base não respondeu ao pedido {pedido['t']!r}")
    if r.get("recusado"):
        raise EnsaioIndisponivelError(f"a base recusou: {r['recusado']}")
    return r


class EnlaceRssi(Ensaio):
    nome = "enlace_rssi"
    fase = FASE_ENLACE
    titulo = "RSSI/SNR, RTT e perda de cada rádio com a base, num ponto (RTK)"
    parametros = (
        Param("ponto", str, None, descricao="nome do ponto do levantamento", obrigatorio=True),
        Param("duracao_s", float, 30.0, 5.0, 3600.0, "quanto tempo medir no ponto"),
        Param("amostra_s", float, 1.0, 0.2, 60.0, "intervalo entre leituras"),
        Param("rssi_min_dbm", float, -85.0, -130.0, 0.0, "RSSI mínimo (mediana)"),
        Param("snr_min_db", float, 15.0, -10.0, 60.0, "SNR mínimo (mediana)"),
        Param("perda_max_pct", float, 10.0, 0.0, 100.0, "perda máxima (média)"),
        Param("rtt_p95_max_ms", float, 300.0, 1.0, 60000.0, "p95 máximo do RTT"),
    )
    hardware_real = HW + " RSSI/SNR só com o adaptador SNMP/HTTP do rádio escolhido."

    def executar(self, ctx: Contexto, p: dict[str, Any]) -> dict[str, Any]:
        f = _fonte(ctx)
        leituras = []
        n = max(1, int(p["duracao_s"] / p["amostra_s"]))
        pos = None
        for i in range(n):
            pos = f.posicao() or pos
            for enlace, s in ((f.vizinhos().get("base") or {}).get("enlaces") or {}).items():
                leituras.append({"i": i, "enlace": enlace, "estado": s.get("estado"),
                                 "rssi_dbm": s.get("rssi_dbm"), "snr_db": s.get("snr_db"),
                                 "perda": s.get("perda"), "rtt_ms": s.get("rtt_ms")})
            ctx.avisa((i + 1) / n, f"leitura {i + 1}/{n}")
            ctx.espera(p["amostra_s"])
        amostras = {"ponto": p["ponto"], "posicao": pos, "leituras": leituras}
        criterios, medidas = avalia_rssi(p, amostras)
        return {"criterios": criterios, "medidas": medidas, "amostras": amostras}


def avalia_rssi(p: dict, a: dict) -> tuple[list[dict], dict]:
    """A mesma conta que a plataforma refaz (``saas/valtexa/services/ensaios.py``)."""
    por: dict[str, dict[str, list]] = {}
    for x in a.get("leituras") or []:
        d = por.setdefault(str(x.get("enlace")), {"rssi": [], "snr": [], "perda": [],
                                                  "rtt": [], "n": [], "vivo": []})
        d["n"].append(1)
        d["vivo"].append(x.get("estado") != "morto")
        for k, c in (("rssi", "rssi_dbm"), ("snr", "snr_db"), ("perda", "perda"),
                     ("rtt", "rtt_ms")):
            v = x.get(c)
            if isinstance(v, int | float) and not isinstance(v, bool):
                d[k].append(float(v))
    medidas: dict[str, Any] = {"ponto": a.get("ponto"), "enlaces": {}}
    melhor, nota = None, None
    for enlace, d in por.items():
        m = {"rssi_p50_dbm": mediana(d["rssi"]) if d["rssi"] else None,
             "snr_p50_db": mediana(d["snr"]) if d["snr"] else None,
             "perda_media_pct": 100 * sum(d["perda"]) / len(d["perda"]) if d["perda"] else
             (100.0 if not any(d["vivo"]) else None),
             "rtt_p95_ms": percentil(d["rtt"], 95) if d["rtt"] else None,
             "vivo_pct": 100 * sum(d["vivo"]) / len(d["vivo"])}
        medidas["enlaces"][enlace] = m
        chave = (m["vivo_pct"], m["rssi_p50_dbm"] if m["rssi_p50_dbm"] is not None else -999,
                 -(m["perda_media_pct"] or 0))
        if nota is None or chave > nota:
            melhor, nota = enlace, chave
    pos = a.get("posicao") or {}
    criterios = [criterio("posicao_rtk", pos.get("fix") in ("fixed", "float"), True, "==")]
    if melhor is None:
        criterios.append(criterio("enlace_com_a_base", None, True, "=="))
        return criterios, medidas
    m = medidas["enlaces"][melhor]
    medidas["melhor_enlace"] = melhor
    criterios += [criterio(f"{melhor}_perda_media_pct", m["perda_media_pct"],
                           p["perda_max_pct"], "<=", "%"),
                  criterio(f"{melhor}_rtt_p95_ms", m["rtt_p95_ms"], p["rtt_p95_max_ms"], "<=",
                           "ms")]
    if m["rssi_p50_dbm"] is not None:
        criterios.append(criterio(f"{melhor}_rssi_p50_dbm", m["rssi_p50_dbm"],
                                  p["rssi_min_dbm"], ">=", "dBm"))
    if m["snr_p50_db"] is not None:
        criterios.append(criterio(f"{melhor}_snr_p50_db", m["snr_p50_db"], p["snr_min_db"],
                                  ">=", "dB"))
    if m["rssi_p50_dbm"] is None:
        medidas["observacao"] = "o rádio não expõe RSSI/SNR: só RTT e perda"
    return criterios, medidas


class EnlaceLatencia(Ensaio):
    nome = "enlace_latencia"
    fase = FASE_ENLACE
    titulo = "Latência de P1 pelo rádio (Rover → base → Rover): p99 e perda"
    parametros = (
        Param("ecos", int, 200, 10, 100000, "idas e voltas"),
        Param("intervalo_s", float, 0.25, 0.0, 60.0, "entre ecos"),
        Param("timeout_s", float, 2.0, 0.1, 30.0, "eco sem resposta neste prazo = perdido"),
        Param("p99_max_ms", float, 500.0, 1.0, 60000.0, "p99 máximo"),
        Param("perda_max_pct", float, 1.0, 0.0, 100.0, "perda máxima"),
    )
    hardware_real = HW

    def executar(self, ctx: Contexto, p: dict[str, Any]) -> dict[str, Any]:
        f = _fonte(ctx)
        rtts: list[float] = []
        for i in range(p["ecos"]):
            r = f.eco(p["timeout_s"])
            if r is not None:
                rtts.append(round(r, 3))
            if (i + 1) % 10 == 0:
                ctx.avisa((i + 1) / p["ecos"], f"eco {i + 1}/{p['ecos']}")
            ctx.espera(p["intervalo_s"])
        amostras = {"ecos": p["ecos"], "rtts_ms": rtts}
        criterios, medidas = avalia_latencia(p, amostras)
        return {"criterios": criterios, "medidas": medidas, "amostras": amostras}


def avalia_latencia(p: dict, a: dict) -> tuple[list[dict], dict]:
    n = int(p["ecos"])
    rtts = [float(x) for x in (a.get("rtts_ms") or [])][:n]
    perda = 100.0 * (n - len(rtts)) / n
    p99 = percentil(rtts, 99) if rtts else None
    medidas = {"p50_ms": round(percentil(rtts, 50), 1) if rtts else None,
               "p95_ms": round(percentil(rtts, 95), 1) if rtts else None,
               "p99_ms": None if p99 is None else round(p99, 1), "perda_pct": round(perda, 2)}
    return [criterio("p99_ms", p99, p["p99_max_ms"], "<=", "ms"),
            criterio("perda_pct", perda, p["perda_max_pct"], "<=", "%")], medidas


class EnlaceTroca(Ensaio):
    nome = "enlace_troca"
    fase = FASE_ENLACE
    titulo = "Troca de enlace: um rádio morre e o tráfego vai pelo outro"
    parametros = (
        Param("enlace", str, None, descricao="o rádio que vai 'morrer' (padrão: o em uso)"),
        Param("modo", str, "base", descricao="'base' (a base silencia por software) ou "
              "'manual' (o técnico desliga o rádio na janela)"),
        Param("silencio_s", float, 20.0, 5.0, 120.0, "quanto tempo o rádio fica mudo"),
        Param("janela_manual_s", float, 120.0, 10.0, 1800.0, "espera do técnico (manual)"),
        Param("eco_s", float, 0.2, 0.05, 5.0, "intervalo dos ecos P1 durante o ensaio"),
        Param("troca_max_s", float, 5.0, 0.1, 120.0, "tempo máximo até o lote trocar"),
        Param("interrupcao_p1_max_s", float, 2.0, 0.0, 120.0, "maior buraco de P1"),
    )
    hardware_real = HW

    def executar(self, ctx: Contexto, p: dict[str, Any]) -> dict[str, Any]:
        f = _fonte(ctx)
        alvo = p["enlace"] or f.melhor()
        if alvo is None:
            raise EnsaioIndisponivelError("nenhum enlace com a base agora")
        if p["modo"] not in ("base", "manual"):
            raise EnsaioIndisponivelError("modo: 'base' ou 'manual'")
        t0 = f.agora()
        t_silencio = None
        if p["modo"] == "base":
            _pede_ou_falha(f, {"t": "silenciar", "enlace": alvo, "s": p["silencio_s"]})
            t_silencio = f.agora()
            fim = t_silencio + p["silencio_s"] + 5.0
        else:
            fim = t0 + p["janela_manual_s"]
        melhores, ecos = [[round(f.agora() - t0, 3), f.melhor()]], []
        ultimo_rx_antes = None
        while f.agora() < fim:
            t = f.agora()
            r = f.eco(min(1.0, p["eco_s"] * 5))
            ecos.append([round(t - t0, 3), None if r is None else round(r, 2)])
            m = f.melhor()
            if m != melhores[-1][1]:
                melhores.append([round(f.agora() - t0, 3), m])
            urx = f.ultimo_rx(alvo)
            if urx is not None and f.agora() - urx > 1.0 and ultimo_rx_antes is None:
                ultimo_rx_antes = urx  # o rádio calou: o instante físico da "morte"
            if p["modo"] == "manual" and ultimo_rx_antes is not None and t_silencio is None:
                t_silencio = ultimo_rx_antes
                fim = min(fim, f.agora() + 30.0)
            ctx.avisa(min(1.0, (f.agora() - t0) / max(1.0, fim - t0)), "medindo a troca")
            ctx.espera(p["eco_s"])
        amostras = {"enlace": alvo, "modo": p["modo"],
                    "t_silencio": None if t_silencio is None else round(t_silencio - t0, 3),
                    "t_ultimo_rx": None if ultimo_rx_antes is None else round(
                        ultimo_rx_antes - t0, 3),
                    "silencio_s": p["silencio_s"], "melhores": melhores, "ecos": ecos}
        criterios, medidas = avalia_troca(p, amostras)
        saida = {"criterios": criterios, "medidas": medidas, "amostras": amostras}
        if amostras["t_silencio"] is None:
            saida["status"] = "nao_avaliado"
            saida["observacao"] = "o rádio não calou dentro da janela: nada para medir"
        return saida


def avalia_troca(p: dict, a: dict) -> tuple[list[dict], dict]:
    t_s = a.get("t_silencio")
    alvo = a.get("enlace")
    if t_s is None:
        return [], {}
    inicio = a.get("t_ultimo_rx") if a.get("t_ultimo_rx") is not None else t_s
    inicio = min(inicio, t_s)
    troca = next((t - inicio for t, m in a.get("melhores") or [] if t >= t_s and m != alvo
                  and m is not None), None)
    fim_janela = t_s + float(a.get("silencio_s") or p["silencio_s"])
    pior, buraco_ini = 0.0, None
    for t, r in a.get("ecos") or []:
        if t < inicio or t > fim_janela:
            continue
        if r is None and buraco_ini is None:
            buraco_ini = t
        elif r is not None and buraco_ini is not None:
            pior = max(pior, t - buraco_ini)
            buraco_ini = None
    if buraco_ini is not None:
        pior = max(pior, fim_janela - buraco_ini)
    medidas = {"enlace_silenciado": alvo, "troca_s": None if troca is None else round(troca, 2),
               "interrupcao_p1_s": round(pior, 2)}
    return [criterio("troca_s", troca, p["troca_max_s"], "<=", "s"),
            criterio("interrupcao_p1_s", pior, p["interrupcao_p1_max_s"], "<=", "s")], medidas


def _sondas_ate_drenar(ctx: Contexto, rede, f: FonteDoEnlace, t0: float, fim_queda: float,
                       p: dict) -> tuple[list, float | None]:
    """Manda uma sonda à nuvem a cada ``sondas_s`` e repete cada uma até entrar (como a
    fila de sincronização). Devolve ``[[criada, entrou]]`` e quando a nuvem voltou."""
    sondas: list[list] = []
    pendentes: list[int] = []
    t_volta = None
    limite = fim_queda + p["recuperacao_max_s"] * 3 + 30
    proxima = f.agora()
    while f.agora() < limite:
        agora = f.agora()
        if agora < fim_queda and agora >= proxima:
            proxima = agora + p["sondas_s"]
            sondas.append([round(agora - t0, 3), None])
            pendentes.append(len(sondas) - 1)
        if t_volta is None and agora >= fim_queda and f.nuvem_ok():
            t_volta = agora
        for i in list(pendentes):
            try:
                ok = rede.ping()
            except Exception:  # sem rede: a sonda continua pendente
                ok = False
            if not ok:
                break
            sondas[i][1] = round(f.agora() - t0, 3)
            pendentes.remove(i)
        if not pendentes and agora >= fim_queda:
            break
        ctx.avisa(min(1.0, (agora - t0) / max(1.0, limite - t0)), "sondas à nuvem")
        ctx.espera(min(1.0, p["sondas_s"]))
    return sondas, t_volta


class EnlaceStarlink(Ensaio):
    nome = "enlace_starlink"
    fase = FASE_ENLACE
    titulo = "Starlink cai e volta: nada se perde e o atraso é entregue"
    parametros = (
        Param("modo", str, "base", descricao="'base' (desliga por software) ou 'manual' "
              "(o técnico tira o cabo da Starlink)"),
        Param("queda_s", float, 60.0, 5.0, 3600.0, "duração da queda"),
        Param("sondas_s", float, 2.0, 0.2, 60.0, "uma sonda à nuvem a cada"),
        Param("recuperacao_max_s", float, 60.0, 1.0, 3600.0,
              "depois que a Starlink volta, tempo máximo até tudo o que ficou entrar"),
    )
    hardware_real = HW + " A volta depende da sonda da base (sonda_s, padrão 10 s)."

    def executar(self, ctx: Contexto, p: dict[str, Any]) -> dict[str, Any]:
        f = _fonte(ctx)
        rede = ctx.fontes.get("rede")
        if rede is None:
            raise EnsaioIndisponivelError("sem canal com a plataforma")
        t0 = f.agora()
        if p["modo"] == "base":
            _pede_ou_falha(f, {"t": "sem_internet", "s": p["queda_s"]})
        fim_queda = f.agora() + p["queda_s"]
        sondas, t_volta = _sondas_ate_drenar(ctx, rede, f, t0, fim_queda, p)
        amostras = {"modo": p["modo"], "t_fim_queda": round(fim_queda - t0, 3),
                    "t_volta_nuvem": None if t_volta is None else round(t_volta - t0, 3),
                    "sondas": sondas}
        criterios, medidas = avalia_starlink(p, amostras)
        return {"criterios": criterios, "medidas": medidas, "amostras": amostras}


def avalia_starlink(p: dict, a: dict) -> tuple[list[dict], dict]:
    sondas = a.get("sondas") or []
    perdidas = sum(1 for _, ok in sondas if ok is None)
    entradas = [ok for _, ok in sondas if ok is not None]
    fim = float(a.get("t_fim_queda") or 0)
    rec = max(entradas) - fim if entradas else None
    if rec is not None:
        rec = max(0.0, rec)
    medidas = {"sondas": len(sondas), "perdidas": perdidas,
               "recuperacao_s": None if rec is None else round(rec, 2),
               "nuvem_voltou_s": a.get("t_volta_nuvem")}
    return [criterio("sondas_perdidas", perdidas, 0, "<="),
            criterio("sondas_feitas", len(sondas), 1, ">="),
            criterio("recuperacao_s", rec, p["recuperacao_max_s"], "<=", "s")], medidas


class EnlaceEnergiaBase(Ensaio):
    nome = "enlace_energia_base"
    fase = FASE_ENLACE
    titulo = "Queda de energia da base: o enlace volta sozinho e nada se perde"
    parametros = (
        Param("modo", str, "base", descricao="'base' (o serviço cai e o systemd sobe) ou "
              "'manual' (o técnico corta a energia da base na janela)"),
        Param("janela_s", float, 300.0, 10.0, 3600.0, "quanto tempo esperar a queda e a volta"),
        Param("eco_s", float, 1.0, 0.1, 60.0, "intervalo dos ecos"),
        Param("sondas_s", float, 2.0, 0.2, 60.0, "uma sonda à nuvem a cada"),
        Param("volta_max_s", float, 120.0, 1.0, 3600.0, "tempo máximo sem enlace"),
        Param("recuperacao_max_s", float, 60.0, 1.0, 3600.0,
              "depois da volta, tempo máximo até as sondas guardadas entrarem"),
    )
    hardware_real = HW + " Com energia de verdade conta o boot do mini PC e o nobreak."

    def executar(self, ctx: Contexto, p: dict[str, Any]) -> dict[str, Any]:
        f = _fonte(ctx)
        rede = ctx.fontes.get("rede")
        t0 = f.agora()
        if p["modo"] == "base":
            _pede_ou_falha(f, {"t": "reiniciar"})
        ecos, sondas, pendentes = [], [], []
        caiu = voltou = None
        fim = t0 + p["janela_s"]
        proxima_sonda = t0
        while f.agora() < fim:
            t = f.agora()
            r = f.eco(min(2.0, max(0.5, p["eco_s"])))
            ecos.append([round(t - t0, 3), None if r is None else round(r, 2)])
            if r is None and caiu is None:
                caiu = t
            if r is not None and caiu is not None and voltou is None:
                voltou = f.agora()
            if rede is not None and voltou is None and f.agora() >= proxima_sonda:
                proxima_sonda = f.agora() + p["sondas_s"]
                sondas.append([round(f.agora() - t0, 3), None])
                pendentes.append(len(sondas) - 1)
            if voltou is not None and rede is not None:
                for i in list(pendentes):
                    try:
                        ok = rede.ping()
                    except Exception:  # ainda sem rede: fica pendente
                        ok = False
                    if not ok:
                        break
                    sondas[i][1] = round(f.agora() - t0, 3)
                    pendentes.remove(i)
            if voltou is not None and not pendentes:
                break
            ctx.avisa(min(1.0, (f.agora() - t0) / p["janela_s"]), "esperando a base")
            ctx.espera(p["eco_s"])
        amostras = {"modo": p["modo"], "ecos": ecos, "sondas": sondas,
                    "t_caiu": None if caiu is None else round(caiu - t0, 3),
                    "t_voltou": None if voltou is None else round(voltou - t0, 3)}
        criterios, medidas = avalia_energia_base(p, amostras)
        saida = {"criterios": criterios, "medidas": medidas, "amostras": amostras}
        if caiu is None:
            saida["status"] = "nao_avaliado"
            saida["observacao"] = "a base não caiu dentro da janela"
        return saida


def avalia_energia_base(p: dict, a: dict) -> tuple[list[dict], dict]:
    caiu, voltou = a.get("t_caiu"), a.get("t_voltou")
    if caiu is None:
        return [], {}
    volta = None if voltou is None else voltou - caiu
    sondas = a.get("sondas") or []
    perdidas = sum(1 for _, ok in sondas if ok is None)
    entradas = [ok for _, ok in sondas if ok is not None]
    rec = None if voltou is None or not entradas else max(0.0, max(entradas) - voltou)
    medidas = {"sem_enlace_s": None if volta is None else round(volta, 2),
               "sondas": len(sondas), "perdidas": perdidas,
               "recuperacao_s": None if rec is None else round(rec, 2)}
    crit = [criterio("sem_enlace_s", volta, p["volta_max_s"], "<=", "s"),
            criterio("sondas_perdidas", perdidas, 0, "<=")]
    if sondas:
        crit.append(criterio("recuperacao_s", rec if rec is not None else (
            0.0 if not entradas and not perdidas else None), p["recuperacao_max_s"], "<=", "s"))
    return crit, medidas


class EnlaceTravessia(Ensaio):
    nome = "enlace_travessia"
    fase = FASE_ENLACE
    titulo = "Travessia de zona morta mapeada (as registradas pelas missões)"
    parametros = (
        Param("ultimas", int, 5, 1, 100, "quantas travessias avaliar"),
        Param("antecedencia_min_s", float, 10.0, 0.0, 600.0, "aviso antes de entrar"),
        Param("sincronizacao_max_s", float, 60.0, 1.0, 3600.0,
              "depois de sair, tempo máximo até sincronizar tudo"),
        Param("folga", float, 1.5, 1.0, 5.0, "duração real até folga x prevista + 15 s"),
    )
    hardware_real = HW + " As travessias vêm das missões (o ensaio não move o robô)."

    def executar(self, ctx: Contexto, p: dict[str, Any]) -> dict[str, Any]:
        f = _fonte(ctx)
        travessias = f.travessias(p["ultimas"])
        amostras = {"travessias": travessias}
        criterios, medidas = avalia_travessia(p, amostras)
        saida = {"criterios": criterios, "medidas": medidas, "amostras": amostras}
        if not travessias:
            saida["status"] = "nao_avaliado"
            saida["observacao"] = ("nenhuma travessia registrada: rode uma missão que cruze "
                                   "uma zona morta do mapa de cobertura")
        return saida


def _seg(iso: str | None) -> float | None:
    from datetime import datetime

    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso).timestamp()
    except ValueError:
        return None


def avalia_travessia(p: dict, a: dict) -> tuple[list[dict], dict]:
    trs = [t for t in a.get("travessias") or [] if isinstance(t, dict)]
    if not trs:
        return [], {}
    antecedencias, syncs, fora = [], [], 0
    for t in trs:
        av, ent = _seg(t.get("avisada_em")), _seg((t.get("real") or {}).get("entrou"))
        if av is not None and ent is not None:
            antecedencias.append(ent - av)
        s = t.get("sincronizado_s_depois_de_sair")
        syncs.append(float(s) if isinstance(s, int | float) else None)
        prev = (t.get("prevista") or {}).get("duracao_s")
        real = (t.get("real") or {}).get("duracao_s")
        if isinstance(prev, int | float) and isinstance(real, int | float) and \
                real > prev * float(p["folga"]) + 15:
            fora += 1
    pior_sync = None if any(s is None for s in syncs) else max(syncs)
    medidas = {"travessias": len(trs),
               "antecedencia_min_s": round(min(antecedencias), 1) if antecedencias else None,
               "sincronizacao_max_s": pior_sync, "fora_do_previsto": fora}
    return [criterio("antecedencia_min_s", min(antecedencias) if antecedencias else None,
                     p["antecedencia_min_s"], ">=", "s"),
            criterio("sincronizacao_max_s", pior_sync, p["sincronizacao_max_s"], "<=", "s"),
            criterio("fora_do_previsto", fora, 0, "<=")], medidas


class Relogio(Ensaio):
    nome = "relogio"
    fase = FASE_ENLACE
    titulo = "Relógio: chrony com GNSS no Rover e na base"
    parametros = (
        Param("offset_max_ms", float, 10.0, 0.001, 60000.0, "|offset| máximo do chrony"),
        Param("exigir_gnss", bool, True, descricao="a fonte em uso tem de ser o GNSS/PPS"),
        Param("atraso_plataforma_max_s", float, 1.0, 0.0, 600.0,
              "quanto o relógio pode estar atrás da hora assinada da plataforma"),
    )
    hardware_real = HW + " O formato do chronyc é o do chrony 4.x (documentação)."

    def executar(self, ctx: Contexto, p: dict[str, Any]) -> dict[str, Any]:
        f = _fonte(ctx)
        rover = f.chrony()
        r = f.pede({"t": "relogio"}, 5.0)
        base = None if r is None else r.get("relogio")
        amostras = {"rover": rover, "base": base, "atraso_plataforma_s": f.atraso_plataforma_s()}
        criterios, medidas = avalia_relogio(p, amostras)
        return {"criterios": criterios, "medidas": medidas, "amostras": amostras}


def avalia_relogio(p: dict, a: dict) -> tuple[list[dict], dict]:
    crit, medidas = [], {}
    for lado in ("rover", "base"):
        r = a.get(lado) or {}
        medidas[lado] = {k: r.get(k) for k in ("sincronizado", "fonte", "offset_ms",
                                               "fonte_em_uso")}
        crit.append(criterio(f"{lado}_sincronizado", r.get("sincronizado"), True, "=="))
        if p["exigir_gnss"]:
            crit.append(criterio(f"{lado}_fonte_gnss", r.get("fonte") == "gnss" if r else None,
                                 True, "=="))
        off = r.get("offset_ms")
        crit.append(criterio(f"{lado}_offset_ms", None if off is None else abs(float(off)),
                             p["offset_max_ms"], "<=", "ms"))
    atraso = a.get("atraso_plataforma_s")
    medidas["atraso_plataforma_s"] = atraso
    if atraso is not None:
        crit.append(criterio("atraso_plataforma_s", float(atraso),
                             p["atraso_plataforma_max_s"], "<=", "s"))
    return crit, medidas


ENSAIOS_DE_ENLACE = (EnlaceRssi, EnlaceLatencia, EnlaceTroca, EnlaceStarlink,
                     EnlaceEnergiaBase, EnlaceTravessia, Relogio)
