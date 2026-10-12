"""Sincronia de relógio por GNSS (chrony): leitura e conferência — base e Rover.

O Rover V3 e a base usam o **chrony** com o GNSS como referência (ZED-F9P: NMEA pelo
``gpsd`` + PPS no pino do Jetson/mini PC). O relógio certo importa para o enlace (validade
dos comandos, piso assinado da hora da plataforma), para a assinatura das requisições (a
plataforma recusa mais de 300 s de diferença) e para casar foto, GNSS e IMU.

:func:`le_chrony` roda ``chronyc -c tracking`` e ``chronyc -c sources`` e
:func:`avalia` diz: sincronizado? por qual fonte (GNSS/PPS, NTP pela internet, nenhuma)?
com que erro? O ensaio ``relogio`` usa isto nos dois lados (a base responde pelo enlace).

Configuração: ``rover/deploy/chrony/valtexa-gnss.conf`` e ``base/deploy/chrony-base.conf``
(``docs/COMUNICACAO_FAZENDA.md`` §11). **Não testado em hardware**: o texto do ``chronyc``
vem da documentação do chrony 4.x; os testes usam saídas gravadas desse formato.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable

# chronyc -c tracking: os campos, na ordem do chrony 4.x
CAMPOS_TRACKING = ("ref_id", "ref_nome", "stratum", "ref_tempo", "offset_s", "ultimo_offset_s",
                   "rms_offset_s", "frequencia_ppm", "freq_residual_ppm", "skew_ppm",
                   "root_delay_s", "root_dispersion_s", "intervalo_s", "leap")
FONTES_GNSS = ("GPS", "PPS", "NMEA", "GNSS", "GPSD", "SHM0", "SHM1", "SOCK")


def parse_tracking(texto: str) -> dict:
    linha = next((x for x in texto.strip().splitlines() if x.strip()), "")
    partes = [p.strip() for p in linha.split(",")]
    if len(partes) < len(CAMPOS_TRACKING):
        raise ValueError(f"chronyc tracking inesperado: {linha!r}")
    d: dict = dict(zip(CAMPOS_TRACKING, partes[:len(CAMPOS_TRACKING)], strict=True))
    for k in ("stratum",):
        d[k] = int(d[k])
    for k in CAMPOS_TRACKING[3:13]:
        d[k] = float(d[k])
    return d


def parse_sources(texto: str) -> list[dict]:
    """``chronyc -c sources``: ``modo,estado,nome,stratum,poll,reach,lastrx,ajustado,medido,
    erro``. Modo ``#`` = relógio de referência local (GNSS/PPS); estado ``*`` = em uso."""
    fontes = []
    for linha in texto.strip().splitlines():
        p = [x.strip() for x in linha.split(",")]
        if len(p) < 10:
            continue
        try:
            fontes.append({"modo": p[0], "estado": p[1], "nome": p[2], "stratum": int(p[3]),
                           "alcance": int(p[5], 8) if p[5] else 0,
                           "offset_s": float(p[8]), "erro_s": float(p[9])})
        except ValueError:
            continue
    return fontes


def le_chrony(executar: Callable | None = None, timeout_s: float = 5.0) -> dict:
    """``{"tracking": {...}, "fontes": [...]}`` ou ``{"erro": "..."}`` (sem chrony)."""
    executar = executar or (lambda args: subprocess.run(
        args, capture_output=True, text=True, timeout=timeout_s, check=True).stdout)
    try:
        return {"tracking": parse_tracking(executar(["chronyc", "-c", "tracking"])),
                "fontes": parse_sources(executar(["chronyc", "-c", "sources"]))}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {"erro": f"chrony indisponível: {exc}"}


def avalia(leitura: dict) -> dict:
    """Resumo: ``sincronizado``, ``fonte`` (``gnss``/``ntp``/``nenhuma``), ``offset_ms``,
    ``erro_estimado_ms`` (offset + raiz), ``stratum``, ``leap``, ``fonte_em_uso``."""
    if "erro" in leitura:
        return {"sincronizado": False, "fonte": "nenhuma", "offset_ms": None,
                "erro_estimado_ms": None, "observacao": leitura["erro"]}
    t = leitura["tracking"]
    em_uso = next((f for f in leitura.get("fontes", []) if f["estado"] == "*"), None)
    fonte_local = em_uso is not None and (
        em_uso["modo"] == "#" or em_uso["nome"].upper().startswith(FONTES_GNSS))
    gnss = fonte_local or str(t["ref_nome"]).upper().startswith(FONTES_GNSS)
    sincronizado = t["leap"].lower() != "not synchronised" and t["stratum"] < 16
    erro = abs(t["offset_s"]) + t["root_delay_s"] / 2 + t["root_dispersion_s"]
    return {"sincronizado": sincronizado,
            "fonte": "gnss" if sincronizado and gnss else "ntp" if sincronizado else "nenhuma",
            "fonte_em_uso": None if em_uso is None else em_uso["nome"],
            "offset_ms": round(t["offset_s"] * 1000, 4),
            "erro_estimado_ms": round(erro * 1000, 4), "stratum": t["stratum"],
            "leap": t["leap"]}
