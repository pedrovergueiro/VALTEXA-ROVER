"""Rede até a plataforma: RTT, perda e vazão — a linha de base para os testes de enlace.

Usa o próprio canal assinado do robô (``/v1/device/ensaio/eco``): ``GET`` sem corpo mede
a ida e volta; ``POST`` com ``carga_kb`` mede a subida; ``GET ?bytes=`` a descida. Uma
requisição que falha (sem enlace, 5xx, tempo esgotado) conta como perda.

A parada de segurança **não depende** disto: este ensaio só diz como está o enlace.
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
from valtexa_rover.ensaios.estatistica import mediana, percentil


class Rede(Ensaio):
    nome = "rede"
    fase = 0
    titulo = "Rede até a plataforma: RTT, perda e vazão"
    parametros = (
        Param("pings", int, 20, 1, 1000, "idas e voltas"),
        Param("intervalo_s", float, 0.5, 0.0, 60.0, "espera entre pings (s)"),
        Param("carga_kb", int, 256, 0, 20480, "tamanho da subida e da descida (KiB); 0 = não"),
        Param("rtt_max_ms", float, 500.0, 1.0, 60000.0, "p95 máximo do RTT (ms)"),
        Param("perda_max_pct", float, 5.0, 0.0, 100.0, "perda máxima (%)"),
        Param("vazao_min_kbps", float, 256.0, 0.0, 1e7,
              "vazão mínima de subida e de descida (kbit/s)"),
    )
    hardware_real = ("simulado/não testado em hardware: a Starlink do Rover no cafezal "
                     "ainda não foi medida.")

    def executar(self, ctx: Contexto, p: dict[str, Any]) -> dict[str, Any]:
        rede = ctx.fontes.get("rede")
        if rede is None:
            raise EnsaioIndisponivelError("sem canal com a plataforma")
        rtts: list[float] = []
        perdidos = 0
        for i in range(p["pings"]):
            t0 = ctx.clock.monotonic()
            try:
                ok = rede.ping()
            except Exception:  # sem enlace conta como perda
                ok = False
            if ok:
                rtts.append((ctx.clock.monotonic() - t0) * 1000)
            else:
                perdidos += 1
            ctx.avisa((i + 1) / (p["pings"] + 2), f"ping {i + 1}/{p['pings']}",
                      {"rtt_ms": round(rtts[-1], 1) if ok else None})
            ctx.espera(p["intervalo_s"])
        perda = 100.0 * perdidos / p["pings"]
        p95 = percentil(rtts, 95) if rtts else None
        criterios = [criterio("perda_pct", perda, p["perda_max_pct"], "<=", "%"),
                     criterio("rtt_p95_ms", p95, p["rtt_max_ms"], "<=", "ms")]
        medidas: dict[str, Any] = {
            "rtt_mediana_ms": round(mediana(rtts), 1) if rtts else None,
            "rtt_p95_ms": None if p95 is None else round(p95, 1),
            "perda_pct": round(perda, 2), "pings": p["pings"]}
        amostras: dict[str, Any] = {"pings": p["pings"], "rtts_ms": [round(r, 3) for r in rtts]}
        if p["carga_kb"] > 0:
            n = p["carga_kb"] * 1024
            for sentido, chamada in (("subida", rede.upload), ("descida", rede.download)):
                t0 = ctx.clock.monotonic()
                try:
                    recebidos = int(chamada(n))
                except Exception:  # sem enlace: vazão zero
                    recebidos = 0
                dt = ctx.clock.monotonic() - t0
                kbps = (8 * recebidos / 1000) / dt if dt > 0 and recebidos else 0.0
                if recebidos != n:
                    kbps = 0.0  # transferência incompleta não é vazão
                medidas[f"vazao_{sentido}_kbps"] = round(kbps, 1)
                amostras[f"{sentido}_bytes"] = recebidos
                amostras[f"{sentido}_s"] = round(dt, 4)
                criterios.append(criterio(f"vazao_{sentido}_kbps", kbps, p["vazao_min_kbps"],
                                          ">=", "kbit/s"))
                ctx.avisa(0.95 if sentido == "subida" else 1.0, sentido)
            amostras["carga_bytes"] = n
        return {"criterios": criterios, "medidas": medidas, "amostras": amostras}
