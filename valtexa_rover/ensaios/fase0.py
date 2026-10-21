"""A FASE 0 inteira (bancada, sem andar), em ordem, e o laudo em Markdown.

Usada por ``python -m valtexa_rover.sim fase0`` (só o robô, prévia local) e pelo
simulador da plataforma (``python -m simulator --fase0``: cada ensaio vai como comando
assinado e o veredito é o do servidor). Na bancada simulada, cada etapa prepara a peça
(ex.: o alvo a 0,3 m para a profundidade) — no robô real quem prepara é o operador.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from valtexa_rover.clock import Clock
from valtexa_rover.hal.interfaces import Hardware

LAT_REF, LON_REF = -21.25, -45.0
"""O "ponto conhecido" da bancada simulada (Sul de Minas)."""


@dataclass
class Etapa:
    rotulo: str
    ensaio: str
    params: dict[str, Any]
    preparo: Callable[[Hardware, Clock], None] | None = None
    instrucao: str = ""
    """O que o operador faz antes, no robô de verdade."""


def _alvo_a(mm: float) -> Callable[[Hardware, Clock], None]:
    def preparo(hardware: Hardware, clock: Clock) -> None:
        for camera in hardware.cameras:
            if hasattr(camera, "distancia_mm"):
                camera.distancia_mm = mm
    return preparo


def _gnss_novo(hardware: Hardware, clock: Clock) -> None:
    """Liga o GNSS da bancada agora (partida a frio) com uma perda na janela do teste."""
    from valtexa_rover.ensaios.simulado import GnssRtkDeBancada

    if isinstance(hardware.gps, GnssRtkDeBancada):
        hardware.gps = GnssRtkDeBancada(clock, LAT_REF, LON_REF, tempo_fix_s=20.0,
                                        perdas=[(60.0, 70.0)])


def etapas_fase0() -> list[Etapa]:
    return [
        Etapa("câmeras (8 juntas)", "cameras", {"duracao_s": 60, "fps": 2},
              instrucao="todas as OAK-D no switch PoE"),
        *[Etapa(f"profundidade {mm / 1000:.1f} m", "profundidade",
                {"camera_id": "oak-0", "distancia_real_mm": mm, "amostras": 10},
                _alvo_a(mm), f"alvo plano a {mm:.0f} mm da oak-0, medido com trena")
          for mm in (300.0, 500.0, 800.0)],
        Etapa("GNSS RTK", "gnss",
              {"lat_ref": LAT_REF, "lon_ref": LON_REF, "duracao_s": 30, "perda_s": 40},
              _gnss_novo, "antena sobre o ponto conhecido; cubra e descubra na janela de perda"),
        Etapa("IMU parada", "imu", {"duracao_s": 30},
              instrucao="robô parado, nivelado (bolha) ou com a inclinação informada"),
        Etapa("rede até a plataforma", "rede", {"pings": 10, "carga_kb": 64}),
        Etapa("relógio (chrony com GNSS)", "relogio", {},
              instrucao="GNSS com fix e PPS ligado; chrony rodando com deploy/chrony/"),
        Etapa("calor (2 h)", "calor", {"duracao_min": 120, "janela_s": 60},
              instrucao="gabinete fechado, modelo de café ativo"),
        Etapa("bateria e energia", "energia", {"duracao_min": 1}),
    ]


def rodar_na_bancada(pasta: Path, *, falhas: dict | None = None,
                     log: Callable[[str], None] = lambda _m: None) -> list[dict[str, Any]]:
    """Toda a FASE 0 na bancada simulada, sem plataforma: os laudos (prévia do robô)."""
    from valtexa_rover.ensaios import Contexto, rodar_por_nome
    from valtexa_rover.ensaios.simulado import RelogioDeBancada, bancada

    relogio = RelogioDeBancada()
    hardware, fontes = bancada(pasta, relogio, falhas=falhas)
    laudos = []
    for etapa in etapas_fase0():
        if etapa.preparo is not None:
            etapa.preparo(hardware, relogio)
        laudo = rodar_por_nome(etapa.ensaio, etapa.params, Contexto(hardware, relogio, fontes))
        laudo["rotulo"] = etapa.rotulo
        log(f"  {laudo['status']:<12} {etapa.rotulo}")
        laudos.append(laudo)
    return laudos


def _valor(c: dict) -> str:
    v = c.get("valor")
    if not isinstance(v, float):
        return f"{v} {c.get('unidade') or ''}".strip()
    texto = f"{v:.0f}" if abs(v) >= 1000 else f"{v:.4g}"
    return f"{texto} {c.get('unidade') or ''}".strip()


def laudo_markdown(linhas: list[dict[str, Any]], *, titulo: str, nota: str = "") -> str:
    """Uma tabela por ensaio: critério, medido, limite, ok. ``linhas`` são laudos (do robô
    ou da plataforma) com ``rotulo``, ``status``, ``criterios`` e ``simulado``."""
    simulado = any(x.get("simulado") for x in linhas)
    partes = [f"# {titulo}", ""]
    if simulado:
        partes += ["> **SIMULADO / não testado em hardware.** Bancada simulada "
                   "(`valtexa_rover.ensaios.simulado`): os números exercitam os ensaios e o "
                   "caminho até o portão; não dizem como o Rover V3 real se comporta.", ""]
    if nota:
        partes += [nota, ""]
    partes += ["| Ensaio | Situação |", "|---|---|"]
    partes += [f"| {x.get('rotulo') or x.get('ensaio')} | {x.get('status')} |" for x in linhas]
    for x in linhas:
        partes += ["", f"## {x.get('rotulo') or x.get('ensaio')} — {x.get('status')}", ""]
        if x.get("criterios"):
            partes += ["| Critério | Medido | Limite | OK |", "|---|---|---|---|"]
            partes += [f"| {c['nome']} | {_valor(c)} | {c.get('op')} {c.get('limite')} | "
                       f"{'sim' if c.get('ok') else 'NÃO'} |" for c in x["criterios"]]
        partes.append("")
        if x.get("erro"):
            partes.append(f"Erro: {x['erro']}")
        if x.get("observacao"):
            partes.append(f"Observação: {x['observacao']}")
    return "\n".join(partes) + "\n"
