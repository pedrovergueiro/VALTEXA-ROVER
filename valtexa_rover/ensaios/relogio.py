"""Ensaio ``relogio`` (FASE 0): o relógio do Rover está certo? chrony com GNSS/PPS.

Até 20/10/2026 ficava na FASE ENLACE, comparando o Rover com a base da fazenda. A base e o
rádio foram arquivados (``archive/legacy/radio-fazenda/``): cada Rover fala com a nuvem pela
própria Starlink. O relógio continua importando — a plataforma recusa requisição assinada
com mais de 300 s de diferença, e foto, GNSS e IMU se casam pela hora —, então o ensaio
ficou, só com o lado do Rover:

* chrony sincronizado, a fonte em uso (GNSS/PPS, NTP pela internet, nenhuma) e o offset;
* o atraso do relógio contra a hora assinada da plataforma (quando o Runtime está rodando).

**Não testado em hardware**: o formato do ``chronyc`` vem da documentação do chrony 4.x.
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
from valtexa_rover.hal import relogio


class FonteDoRelogio:
    """O que o ensaio lê: o chrony desta máquina e, com o Runtime, a hora da plataforma."""

    def __init__(self, core=None, le_chrony=None) -> None:
        self.core = core
        self.le_chrony = le_chrony or relogio.le_chrony

    def chrony(self) -> dict:
        return relogio.avalia(self.le_chrony())

    def atraso_plataforma_s(self) -> float | None:
        if self.core is None or not hasattr(self.core, "piso_plataforma"):
            return None
        return self.core.piso_plataforma.atraso_s(self.core.clock.now())


class Relogio(Ensaio):
    nome = "relogio"
    fase = 0
    titulo = "Relógio: chrony com GNSS no Rover"
    parametros = (
        Param("offset_max_ms", float, 10.0, 0.001, 60000.0, "|offset| máximo do chrony"),
        Param("exigir_gnss", bool, True, descricao="a fonte em uso tem de ser o GNSS/PPS"),
        Param("atraso_plataforma_max_s", float, 1.0, 0.0, 600.0,
              "quanto o relógio pode estar atrás da hora assinada da plataforma"),
    )
    hardware_real = ("não testado em hardware: o formato do chronyc é o do chrony 4.x "
                     "(documentação).")

    def executar(self, ctx: Contexto, p: dict[str, Any]) -> dict[str, Any]:
        f = ctx.fontes.get("relogio")
        if f is None:
            raise EnsaioIndisponivelError("sem leitura do relógio (chrony) neste ambiente")
        amostras = {"rover": f.chrony(), "atraso_plataforma_s": f.atraso_plataforma_s()}
        criterios, medidas = avalia_relogio(p, amostras)
        return {"criterios": criterios, "medidas": medidas, "amostras": amostras}


def avalia_relogio(p: dict, a: dict) -> tuple[list[dict], dict]:
    """A conta do laudo (a plataforma refaz a mesma em ``services/ensaios.py``)."""
    r = a.get("rover") or {}
    medidas: dict[str, Any] = {"rover": {k: r.get(k) for k in ("sincronizado", "fonte",
                                                              "offset_ms", "fonte_em_uso")}}
    crit = [criterio("rover_sincronizado", r.get("sincronizado"), True, "==")]
    if p["exigir_gnss"]:
        crit.append(criterio("rover_fonte_gnss", r.get("fonte") == "gnss" if r else None,
                             True, "=="))
    off = r.get("offset_ms")
    crit.append(criterio("rover_offset_ms", None if off is None else abs(float(off)),
                         p["offset_max_ms"], "<=", "ms"))
    atraso = a.get("atraso_plataforma_s")
    medidas["atraso_plataforma_s"] = atraso
    if atraso is not None:
        crit.append(criterio("atraso_plataforma_s", float(atraso),
                             p["atraso_plataforma_max_s"], "<=", "s"))
    return crit, medidas
