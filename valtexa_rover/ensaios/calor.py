"""Calor: N minutos de inferência sem parar, com o ``tegrastats`` ao lado.

Mede a vazão (quadros/s) em janelas e compara a última com a primeira: a Jetson que
esquenta baixa o clock (*throttling*) e a vazão cai. Lê do ``tegrastats`` a temperatura
mais alta, os clocks de CPU e GPU e a ocupação da GPU, a cada segundo.

Aprova se a vazão da última janela não cair mais que ``perda_max_pct`` em relação à
primeira **e** nenhum clock cair mais que ``queda_clock_pct`` do máximo visto com carga
(o sinal de *throttling* que o ``tegrastats`` mostra). A temperatura máxima vai no laudo
e é critério só se ``temp_max_c`` for dado.

Fontes (``Contexto.fontes``): ``carga`` — uma chamada = uma inferência, devolve quantos
quadros processou; ``termica`` — ``ler()`` com ``temp_c``, ``cpu_freq_mhz``,
``gpu_freq_mhz``, ``gpu_pct`` (o que faltar sai ``None``).
"""

from __future__ import annotations

from typing import Any

from valtexa_rover.ensaios.base import (
    Contexto,
    Ensaio,
    EnsaioIndisponivelError,
    Param,
    criterio,
)


def queda_de_clock_pct(serie: list[float | None]) -> float | None:
    """Maior queda (%) de um clock em relação ao máximo visto **antes** dela."""
    pico, pior = None, 0.0
    for v in serie:
        if v is None:
            continue
        if pico is None or v > pico:
            pico = v
        elif pico > 0:
            pior = max(pior, 100.0 * (pico - v) / pico)
    return None if pico is None else pior


class Calor(Ensaio):
    nome = "calor"
    fase = 0
    titulo = "Calor: inferência contínua sem perder vazão"
    parametros = (
        Param("duracao_min", float, 120.0, 0.05, 720.0, "minutos de carga contínua"),
        Param("janela_s", float, 60.0, 0.1, 3600.0, "janela da vazão (s)"),
        Param("amostra_s", float, 1.0, 0.1, 60.0, "intervalo de leitura do tegrastats (s)"),
        Param("perda_max_pct", float, 10.0, 0.0, 100.0,
              "queda máxima da vazão, última janela contra a primeira (%)"),
        Param("queda_clock_pct", float, 20.0, 0.0, 100.0,
              "queda de clock que conta como throttling (%)"),
        Param("temp_max_c", float, None, 20.0, 120.0,
              "temperatura máxima aceita (°C); vazio = só registra"),
    )
    hardware_real = ("simulado/não testado em hardware: o formato do tegrastats do JetPack 6 "
                     "e a vazão real do modelo de café na Orin NX ainda não foram medidos.")

    def executar(self, ctx: Contexto, p: dict[str, Any]) -> dict[str, Any]:
        carga = ctx.fontes.get("carga")
        if carga is None:
            raise EnsaioIndisponivelError("sem carga de inferência (nenhum modelo ativo)")
        termica = ctx.fontes.get("termica")
        duracao = p["duracao_min"] * 60.0
        janela = min(p["janela_s"], duracao)
        inicio = ctx.clock.monotonic()
        janelas: list[float] = []  # quadros/s de cada janela fechada
        leituras: list[dict[str, Any]] = []
        quadros_janela, inicio_janela = 0, inicio
        ultima_leitura = -1e18
        while True:
            agora = ctx.clock.monotonic()
            decorrido = agora - inicio
            if agora - ultima_leitura >= p["amostra_s"] and termica is not None:
                ultima_leitura = agora
                bruto = dict(termica.ler() or {})
                leituras.append({"t": round(decorrido, 2), **{
                    k: bruto.get(k) for k in ("temp_c", "cpu_freq_mhz", "gpu_freq_mhz",
                                              "gpu_pct")}})
            if agora - inicio_janela >= janela:
                janelas.append(quadros_janela / (agora - inicio_janela))
                quadros_janela, inicio_janela = 0, agora
                ctx.avisa(decorrido / duracao, f"janela {len(janelas)}",
                          {"vazao_qps": round(janelas[-1], 3),
                           "temp_c": leituras[-1].get("temp_c") if leituras else None})
            if decorrido >= duracao:
                break
            ctx.checa()
            quadros_janela += int(carga() or 0)
        if not janelas:
            raise EnsaioIndisponivelError("nenhuma janela fechou: aumente a duração")
        primeira, ultima = janelas[0], janelas[-1]
        perda = 100.0 * (primeira - ultima) / primeira if primeira > 0 else 100.0
        temps = [x["temp_c"] for x in leituras if x.get("temp_c") is not None]
        quedas = [queda_de_clock_pct([x.get(k) for x in leituras])
                  for k in ("cpu_freq_mhz", "gpu_freq_mhz")]
        quedas_medidas = [q for q in quedas if q is not None]
        criterios = [criterio("perda_vazao_pct", perda, p["perda_max_pct"], "<=", "%")]
        if quedas_medidas:
            criterios.append(criterio("queda_clock_pct", max(quedas_medidas),
                                      p["queda_clock_pct"], "<=", "%"))
        if p["temp_max_c"] is not None:
            criterios.append(criterio("temp_max_c", max(temps) if temps else None,
                                      p["temp_max_c"], "<=", "°C"))
        medidas = {
            "vazao_primeira_qps": round(primeira, 3), "vazao_ultima_qps": round(ultima, 3),
            "vazao_min_qps": round(min(janelas), 3), "perda_vazao_pct": round(perda, 2),
            "janelas": len(janelas),
            "temp_max_c": max(temps) if temps else None,
            "temp_inicial_c": temps[0] if temps else None,
            "queda_clock_cpu_pct": None if quedas[0] is None else round(quedas[0], 1),
            "queda_clock_gpu_pct": None if quedas[1] is None else round(quedas[1], 1),
            "tegrastats": termica is not None,
        }
        return {"criterios": criterios, "medidas": medidas,
                "amostras": {"janelas_qps": [round(j, 4) for j in janelas],
                             "termica": leituras[-3600:]}}
