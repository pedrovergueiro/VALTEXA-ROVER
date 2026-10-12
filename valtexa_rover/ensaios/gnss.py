"""GNSS RTK: tempo até o fix fixo, precisão sobre ponto conhecido, perda e retomada.

Três etapas, nesta ordem:

1. **Tempo até fix RTK** — lê o receptor a ``taxa_hz`` até o fix ser ``fixed``
   (qualidade GGA 4). Liga o ensaio com o receptor recém-ligado (ou a correção recém-
   conectada) para medir a partida a frio.
2. **Precisão** — ``duracao_s`` segundos de posições fixas com a antena parada sobre um
   ponto conhecido (``lat_ref``/``lon_ref``, levantado por outro meio) ou, sem ele, em
   torno da própria média (aí só a dispersão; o viés fica sem medir). CEP50, CEP95 e
   2DRMS em cm.
3. **Perda e retomada** (se ``perda_s`` > 0) — durante ``perda_s`` segundos o operador
   cobre a antena (ou desliga a correção) e descobre; o ensaio anota cada perda do fix
   fixo e quanto tempo levou para voltar. Sem nenhuma perda na janela, reprova: o teste
   não foi feito.
"""

from __future__ import annotations

from typing import Any

from valtexa_rover.ensaios.base import Contexto, Ensaio, Param, criterio
from valtexa_rover.ensaios.estatistica import precisao_horizontal


def _le(ctx: Contexto):
    try:
        return ctx.hardware.gps.read()
    except Exception:  # sem posição conta como "sem fix"
        return None


class Gnss(Ensaio):
    nome = "gnss"
    fase = 0
    titulo = "GNSS RTK: tempo até fix, precisão, perda e retomada"
    parametros = (
        Param("tempo_fix_max_s", float, 120.0, 1.0, 3600.0, "espera máxima pelo fix RTK (s)"),
        Param("duracao_s", float, 60.0, 1.0, 7200.0, "coleta de precisão (s)"),
        Param("taxa_hz", float, 5.0, 0.2, 20.0, "leituras por segundo"),
        Param("lat_ref", float, None, -90.0, 90.0, "latitude do ponto conhecido"),
        Param("lon_ref", float, None, -180.0, 180.0, "longitude do ponto conhecido"),
        Param("cep95_max_cm", float, 5.0, 0.1, 10000.0, "CEP95 máximo (cm)"),
        Param("vies_max_cm", float, 5.0, 0.1, 10000.0,
              "viés máximo até o ponto conhecido (cm)"),
        Param("perda_s", float, 0.0, 0.0, 3600.0,
              "janela do teste de perda e retomada (s); 0 = não faz"),
        Param("retomada_max_s", float, 60.0, 1.0, 3600.0, "tempo máximo para voltar ao fixo (s)"),
    )
    hardware_real = ("simulado/não testado em hardware: ZED-F9P com correção NTRIP/base "
                     "ainda não ligado; o tempo até fix depende da correção da fazenda.")

    def executar(self, ctx: Contexto, p: dict[str, Any]) -> dict[str, Any]:
        intervalo = 1.0 / p["taxa_hz"]
        total = p["tempo_fix_max_s"] + p["duracao_s"] + p["perda_s"]
        inicio = ctx.clock.monotonic()
        # 1. tempo até o fix fixo
        tempo_fix = None
        fixes_vistos: dict[str, int] = {}
        while ctx.clock.monotonic() - inicio <= p["tempo_fix_max_s"]:
            fix = _le(ctx)
            tipo = fix.fix if fix is not None else "sem_posicao"
            fixes_vistos[tipo] = fixes_vistos.get(tipo, 0) + 1
            if tipo == "fixed":
                tempo_fix = ctx.clock.monotonic() - inicio
                break
            ctx.avisa((ctx.clock.monotonic() - inicio) / total, "esperando fix RTK",
                      {"fix": tipo})
            ctx.espera(intervalo)
        criterios = [criterio("tempo_fix_s", tempo_fix, p["tempo_fix_max_s"], "<=", "s")]
        medidas: dict[str, Any] = {"tempo_fix_s": None if tempo_fix is None
                                   else round(tempo_fix, 2), "fixes_vistos": fixes_vistos}
        amostras: dict[str, Any] = {"tempo_fix_s": tempo_fix}
        if tempo_fix is None:
            medidas["observacao"] = "sem fix RTK no prazo: precisão e perda não medidas"
            return {"criterios": criterios, "medidas": medidas, "amostras": amostras}

        # 2. precisão
        lats, lons, lidas, fixas = [], [], 0, 0
        t_prec = ctx.clock.monotonic()
        while ctx.clock.monotonic() - t_prec < p["duracao_s"]:
            fix = _le(ctx)
            lidas += 1
            if fix is not None and fix.fix == "fixed":
                fixas += 1
                lats.append(fix.lat)
                lons.append(fix.lon)
            ctx.avisa((ctx.clock.monotonic() - inicio) / total, "medindo precisão",
                      {"amostras": len(lats)})
            ctx.espera(intervalo)
        ref = None
        if p["lat_ref"] is not None and p["lon_ref"] is not None:
            ref = (p["lat_ref"], p["lon_ref"])
        if lats:
            prec = precisao_horizontal(lats, lons, ref)
            medidas.update({k: round(v, 2) for k, v in prec.items()})
        else:
            prec = {}
        medidas["fracao_fixa"] = round(fixas / lidas, 4) if lidas else 0.0
        medidas["referencia"] = "ponto conhecido" if ref else "média das posições (sem viés)"
        criterios.append(criterio("cep95_cm", prec.get("cep95_cm"), p["cep95_max_cm"], "<=",
                                  "cm"))
        if ref is not None:
            criterios.append(criterio("vies_cm", prec.get("vies_cm"), p["vies_max_cm"], "<=",
                                      "cm"))
        amostras.update({"lat": lats[-20000:], "lon": lons[-20000:],
                         "ref": list(ref) if ref else None})

        # 3. perda e retomada
        if p["perda_s"] > 0:
            perdas: list[dict[str, Any]] = []
            caiu_em = None
            t_perda = ctx.clock.monotonic()
            while ctx.clock.monotonic() - t_perda < p["perda_s"]:
                fix = _le(ctx)
                agora = ctx.clock.monotonic() - t_perda
                fixo = fix is not None and fix.fix == "fixed"
                if not fixo and caiu_em is None:
                    caiu_em = agora
                    perdas.append({"inicio_s": round(agora, 2), "retomada_s": None,
                                   "pior": fix.fix if fix is not None else "sem_posicao"})
                elif fixo and caiu_em is not None:
                    perdas[-1]["retomada_s"] = round(agora - caiu_em, 2)
                    caiu_em = None
                ctx.avisa((ctx.clock.monotonic() - inicio) / total,
                          "perda e retomada: cubra e descubra a antena",
                          {"perdas": len(perdas), "sem_fix": caiu_em is not None})
                ctx.espera(intervalo)
            retomadas = [x["retomada_s"] for x in perdas if x["retomada_s"] is not None]
            sem_volta = sum(1 for x in perdas if x["retomada_s"] is None)
            criterios.append(criterio("perdas_observadas", len(perdas), 1, ">="))
            criterios.append(criterio("perdas_sem_retomada", sem_volta, 0, "=="))
            criterios.append(criterio("retomada_max_s", max(retomadas) if retomadas else None,
                                      p["retomada_max_s"], "<=", "s"))
            medidas["perdas"] = perdas
            amostras["perdas"] = perdas
        return {"criterios": criterios, "medidas": medidas, "amostras": amostras}
