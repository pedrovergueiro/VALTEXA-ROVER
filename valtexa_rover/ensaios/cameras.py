"""Câmeras: de 1 a 8 fotografando juntas — perda de quadro, deriva entre os carimbos de
tempo e sincronia com o GNSS.

A cada rodada (``fps`` por segundo) dispara todas as câmeras escolhidas **ao mesmo tempo**
(uma thread por câmera, como no switch PoE) e anota, por câmera, se a foto veio e o
carimbo dela; lê o GNSS no mesmo instante.

* **perda**: captura que falhou + salto no ``frame_index`` (quadro que a câmera contou
  e não entregou), sobre o esperado;
* **deriva**: em cada rodada, o carimbo mais novo menos o mais velho entre as câmeras
  (ms) — p95 e máximo;
* **sincronia com o GNSS**: idade da posição do GNSS no carimbo de cada foto (ms). Com o
  robô a 1 m/s, 100 ms são 10 cm ao longo da fileira.

Os carimbos são os do Runtime (relógio da Jetson). Sincronia de hardware (PPS do GNSS
nas OAK-D) não é medida aqui — simulado/não testado em hardware.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

from valtexa_rover.ensaios.base import (
    Contexto,
    Ensaio,
    EnsaioIndisponivelError,
    Param,
    criterio,
)
from valtexa_rover.ensaios.estatistica import percentil


def _captura(camera) -> tuple[bool, datetime | None, int | None, str | None]:
    try:
        q = camera.capture()
        return True, q.captured_at, q.frame_index, None
    except Exception as exc:  # câmera que falha conta como perda, não para o ensaio
        return False, None, None, f"{type(exc).__name__}: {exc}"


class Cameras(Ensaio):
    nome = "cameras"
    fase = 0
    titulo = "Câmeras simultâneas: perda, deriva e sincronia com o GNSS"
    parametros = (
        Param("cameras", int, None, 1, 8, "quantas câmeras (as primeiras do hardware.json); "
                                          "vazio = todas"),
        Param("duracao_s", float, 60.0, 1.0, 7200.0, "duração (s)"),
        Param("fps", float, 2.0, 0.1, 30.0, "rodadas por segundo"),
        Param("perda_max_pct", float, 0.0, 0.0, 100.0, "perda máxima de quadros (%)"),
        Param("deriva_max_ms", float, 100.0, 0.0, 10000.0,
              "p95 da deriva entre câmeras numa rodada (ms)"),
        Param("gnss_max_ms", float, 200.0, 0.0, 10000.0,
              "p95 da idade do GNSS no carimbo da foto (ms)"),
    )
    hardware_real = ("simulado/não testado em hardware: 8 OAK-D Pro W PoE no mesmo switch "
                     "e o PPS do ZED-F9P ainda não foram ligados juntos.")

    def executar(self, ctx: Contexto, p: dict[str, Any]) -> dict[str, Any]:
        todas = list(ctx.hardware.cameras)
        n = p["cameras"] or len(todas)
        if n > len(todas):
            raise EnsaioIndisponivelError(f"pediu {n} câmeras e o rover tem {len(todas)}")
        cameras = todas[:n]
        ids = [c.camera_id for c in cameras]
        rodadas = max(1, round(p["duracao_s"] * p["fps"]))
        intervalo = 1.0 / p["fps"]
        falhas = dict.fromkeys(ids, 0)
        saltos = dict.fromkeys(ids, 0)
        ultimo_indice: dict[str, int | None] = dict.fromkeys(ids)
        erros: dict[str, str] = {}
        derivas_ms: list[float] = []
        idades_gnss_ms: list[float] = []
        sem_gnss = 0
        with ThreadPoolExecutor(max_workers=len(cameras)) as pool:
            for r in range(rodadas):
                inicio = ctx.clock.monotonic()
                resultados = list(pool.map(_captura, cameras))
                try:
                    fix = ctx.hardware.gps.read()
                except Exception:  # GNSS fora: conta e segue
                    fix = None
                carimbos = []
                for cid, (ok, quando, indice, erro) in zip(ids, resultados, strict=True):
                    if not ok:
                        falhas[cid] += 1
                        erros[cid] = erro or "falha"
                        continue
                    carimbos.append(quando)
                    anterior = ultimo_indice[cid]
                    if indice is not None and anterior is not None and indice > anterior + 1:
                        saltos[cid] += indice - anterior - 1
                    ultimo_indice[cid] = indice
                    if fix is not None:
                        idades_gnss_ms.append(
                            abs((quando - fix.recorded_at).total_seconds()) * 1000)
                    else:
                        sem_gnss += 1
                if len(carimbos) >= 2:
                    derivas_ms.append((max(carimbos) - min(carimbos)).total_seconds() * 1000)
                elif len(carimbos) == 1:
                    derivas_ms.append(0.0)
                ctx.avisa((r + 1) / rodadas, f"rodada {r + 1}/{rodadas}",
                          {"falhas": sum(falhas.values()), "saltos": sum(saltos.values())})
                ctx.espera(intervalo - (ctx.clock.monotonic() - inicio))
        esperadas = rodadas * len(cameras)
        perdidas = sum(falhas.values()) + sum(saltos.values())
        perda_pct = 100.0 * perdidas / esperadas
        deriva_p95 = percentil(derivas_ms, 95) if derivas_ms else None
        gnss_p95 = percentil(idades_gnss_ms, 95) if idades_gnss_ms else None
        criterios = [
            criterio("perda_pct", perda_pct, p["perda_max_pct"], "<=", "%"),
            criterio("deriva_p95_ms", deriva_p95, p["deriva_max_ms"], "<=", "ms"),
            criterio("gnss_p95_ms", gnss_p95, p["gnss_max_ms"], "<=", "ms"),
        ]
        por_camera = {cid: {"falhas": falhas[cid], "saltos": saltos[cid],
                            "erro": erros.get(cid)} for cid in ids}
        medidas = {
            "cameras": len(cameras), "rodadas": rodadas, "esperadas": esperadas,
            "perdidas": perdidas, "perda_pct": round(perda_pct, 3),
            "deriva_p95_ms": None if deriva_p95 is None else round(deriva_p95, 2),
            "deriva_max_ms": round(max(derivas_ms), 2) if derivas_ms else None,
            "gnss_p95_ms": None if gnss_p95 is None else round(gnss_p95, 2),
            "fotos_sem_gnss": sem_gnss, "por_camera": por_camera,
        }
        return {"criterios": criterios, "medidas": medidas,
                "amostras": {"cameras": len(cameras), "esperadas": esperadas,
                             "perdidas": perdidas,
                             "derivas_ms": [round(d, 3) for d in derivas_ms[-5000:]],
                             "idades_gnss_ms": [round(d, 3) for d in idades_gnss_ms[-5000:]]}}
