"""Bateria e energia: interface pronta, **aguarda protocolo do BMS/MCU**.

A bateria (48 V, 100 Ah) e o consumo vêm do microcontrolador de segurança, cujo protocolo
ainda não existe. Este ensaio já tem a forma final: lê a fonte ``energia`` (``ler()`` →
``tensao_v``, ``corrente_a``, ``soc_pct``; o que faltar, ``None``) por ``duracao_min``
minutos, integra a energia e estima a autonomia contra as 8 h de projeto.

Sem fonte de energia, o laudo sai ``nao_avaliado`` com "aguarda protocolo do BMS/MCU" —
nunca um número inventado. Com os sensores simulados (só ``battery_percent``), estima a
autonomia pela queda do SOC e o laudo diz que é simulação.
"""

from __future__ import annotations

from typing import Any

from valtexa_rover.ensaios.base import Contexto, Ensaio, Param, criterio

AGUARDA = "aguarda protocolo do BMS/MCU"


class SensoresComoEnergia:
    """Adapta os ``Sensors`` da HAL (``battery_percent``, ``battery_voltage_v``,
    ``battery_current_a``) à fonte ``energia``."""

    def __init__(self, sensors) -> None:
        self.sensors = sensors

    def ler(self) -> dict[str, float | None]:
        v = self.sensors.read().values
        return {"soc_pct": v.get("battery_percent"), "tensao_v": v.get("battery_voltage_v"),
                "corrente_a": v.get("battery_current_a")}


class Energia(Ensaio):
    nome = "energia"
    fase = 0
    titulo = "Bateria e energia (aguarda protocolo do BMS/MCU)"
    parametros = (
        Param("duracao_min", float, 10.0, 0.05, 600.0, "minutos de leitura"),
        Param("amostra_s", float, 5.0, 0.1, 600.0, "intervalo de leitura (s)"),
        Param("autonomia_min_h", float, 8.0, 0.1, 100.0, "autonomia mínima estimada (h)"),
    )
    hardware_real = AGUARDA

    def executar(self, ctx: Contexto, p: dict[str, Any]) -> dict[str, Any]:
        fonte = ctx.fontes.get("energia")
        if fonte is None:
            return {"status": "nao_avaliado", "observacao": AGUARDA,
                    "medidas": {"fonte": None}, "amostras": {}}
        duracao = p["duracao_min"] * 60
        inicio = ctx.clock.monotonic()
        leituras: list[dict[str, Any]] = []
        while True:
            t = ctx.clock.monotonic() - inicio
            bruto = fonte.ler() or {}
            leituras.append({"t": round(t, 2), **{k: bruto.get(k) for k in
                                                   ("tensao_v", "corrente_a", "soc_pct")}})
            ctx.avisa(t / duracao, "lendo a bateria", {"soc_pct": bruto.get("soc_pct")})
            if t >= duracao:
                break
            ctx.espera(min(p["amostra_s"], duracao - t))
        potencias = [x["tensao_v"] * x["corrente_a"] for x in leituras
                     if x["tensao_v"] is not None and x["corrente_a"] is not None]
        socs = [(x["t"], x["soc_pct"]) for x in leituras if x["soc_pct"] is not None]
        autonomia = None
        if len(socs) >= 2 and socs[-1][0] > socs[0][0]:
            queda_por_h = (socs[0][1] - socs[-1][1]) / ((socs[-1][0] - socs[0][0]) / 3600)
            if queda_por_h > 0:
                autonomia = socs[0][1] / queda_por_h
        medidas = {
            "potencia_media_w": round(sum(potencias) / len(potencias), 1) if potencias else None,
            "soc_inicial_pct": socs[0][1] if socs else None,
            "soc_final_pct": socs[-1][1] if socs else None,
            "autonomia_estimada_h": None if autonomia is None else round(autonomia, 2),
            "leituras": len(leituras),
        }
        if autonomia is None:
            return {"status": "nao_avaliado", "observacao": f"sem queda de SOC medida; {AGUARDA}",
                    "medidas": medidas, "amostras": {"leituras": leituras}}
        return {"criterios": [criterio("autonomia_h", autonomia, p["autonomia_min_h"], ">=",
                                       "h")],
                "medidas": medidas, "amostras": {"leituras": leituras[-5000:]},
                "observacao": AGUARDA}
