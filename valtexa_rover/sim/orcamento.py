"""SIMULAÇÃO: quanto um Rover manda pela Starlink num dia, antes e depois do orçamento.

``python -m valtexa_rover.sim.orcamento`` imprime a tabela de ``docs/ROVER_RUNTIME.md``
(orçamento de dados). **Tudo aqui é simulado**: o fluxo de quadros, a taxa de evento e os
tamanhos de arquivo são hipóteses declaradas em :class:`Hipoteses`; o que é **código de
verdade** é a política — o :class:`~valtexa_rover.runtime.orcamento.DetectorDeTempestade`,
o :class:`~valtexa_rover.runtime.orcamento.DeduplicadorPorPlanta`, as frações de corte por
prioridade de :class:`~valtexa_rover.config.DataBudgetConfig` e as regras de retenção
padrão de :class:`~valtexa_rover.config.RetentionConfig` (o que vira original, reduzida,
contexto do buffer).

O dia: 8 câmeras (4 de cada lado, alturas diferentes), 1 quadro/s, 10 h, Rover a
``velocidade_ms`` na entrelinha. A planta (rua, índice) sai da posição, como no Rover com a
navegação. A fila esvazia na ordem de prioridade, limitada pela banda de subida; o teto
corta pela fração de cada prioridade (como o :class:`SyncEngine`). Resultado: bytes por
dia — o que subiu, o que ficou esperando o teto e o que ficou retido no disco.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field

from valtexa_rover.config import DataBudgetConfig, Priority, RetentionConfig
from valtexa_rover.runtime.orcamento import DeduplicadorPorPlanta, DetectorDeTempestade


@dataclass(frozen=True)
class Hipoteses:
    """Os números que não foram medidos em campo (todos **hipótese**)."""

    cameras_por_lado: int = 4
    horas: float = 10.0
    fps: float = 1.0
    velocidade_ms: float = 0.3
    """Inspeção devagar na entrelinha do café."""
    espacamento_planta_m: float = 0.7
    campo_ao_longo_m: float = 0.6
    """Quanto da rua cada foto cobre a ~27 cm da saia (cada planta aparece em ~2 quadros
    seguidos de cada câmera)."""
    original_bytes: int = 2_500_000
    """Foto parada de 12 MP em JPEG do encoder da OAK-D, de folhagem: hipótese (~0,2 B/px)."""
    reduzida_bytes: int = 12_000
    """320 px, JPEG 60 (a regra NORMAL)."""
    contexto_bytes: int = 180_000
    """1024 px, JPEG 85 (o buffer)."""
    registro_resultado_bytes: int = 12_786
    """JSON canônico de um resultado com 70 recortes e 6 classes, ``tile_probabilities:
    full`` (:func:`tamanho_do_resultado`, medido no corpo sintético)."""
    registro_evento_bytes: int = 900
    telemetria_bytes_por_amostra: int = 2_500
    telemetria_intervalo_s: float = 10.0
    registros_por_requisicao: int = 200
    chunk_bytes: int = 256 * 1024
    sobrecarga_requisicao_bytes: int = 700
    banda_subida_bps: float = 8e6
    """Subida da Starlink no campo, sustentada: hipótese (8 Mbit/s)."""
    baixa_confianca: float = 0.02
    """Fração dos quadros NORMAL marcada baixa confiança (sobe em ORIGINAL pela regra)."""
    vistas_extras_por_validacao: float = 1.5
    """Vistas extras originais que a percepção ativa guarda por quadro na dúvida."""
    fracao_para_validacao: float = 0.01


@dataclass(frozen=True)
class Cenario:
    nome: str
    infectadas: float
    """Fração das plantas com sinal (evento quando a câmera a vê)."""
    deteccao: float = 0.8
    alarme_falso: float = 0.005
    tempestade: bool = False
    """O modelo dispara em todo quadro (visto em vídeo real: 511/511)."""


CENARIOS = (
    Cenario("normal", 0.02),
    Cenario("alta", 0.10),
    Cenario("tempestade", 0.10, tempestade=True),
)


@dataclass
class Balanco:
    gerado: dict[str, int] = field(default_factory=dict)
    subiu: dict[str, int] = field(default_factory=dict)
    esperando: int = 0
    retido: int = 0
    retido_itens: dict[str, int] = field(default_factory=dict)

    def soma(self, d: dict[str, int], chave: str, n: int) -> None:
        d[chave] = d.get(chave, 0) + int(n)

    @property
    def total_gerado(self) -> int:
        return sum(self.gerado.values())

    @property
    def total_subiu(self) -> int:
        return sum(self.subiu.values())


def corpo_do_resultado(recortes: int = 70, classes: int = 6, seed: int = 0,
                       modo: str = "full") -> dict:
    """Um resultado típico (a forma do ``RoverCore._result_body``), com probabilidades de
    recorte em ``float`` (o que o modelo devolve). ``modo``: o ``tile_probabilities`` do
    perfil — ``full`` (todas as classes de cada recorte) ou ``top`` (só a líder)."""
    rng = random.Random(seed)
    rotulos = ["Healthy", "Rust", "Cercospora", "Phoma", "Leaf_Miner_Damage",
               "Unknown_Background"][:classes]

    def probs():
        v = [rng.random() for _ in rotulos]
        s = sum(v)
        return [x / s for x in v]

    corpo = {
        "id": "8c1f0f7e-6a5b-4c1a-9f7e-2b6f2f0a1c3d", "kind": "result", "type": "NORMAL",
        "device_id": "a" * 32, "mission_id": "b" * 32, "field_id": "c" * 32,
        "occurred_at": "2026-10-18T13:00:00.000000+00:00",
        "navigation": {"row_left": "L12", "row_right": "L13", "x_m": 123.45},
        "frame": {"camera_id": "cam-esq-2", "frame_index": 12345, "width": 4056,
                  "height": 3040, "content_type": "image/jpeg", "view": None},
        "position": {"lat": -21.12345678, "lon": -45.12345678, "alt_m": 912.3,
                     "fix": "fixed", "hdop": 0.6, "heading_deg": 87.2},
        "model": {"key": "cafe-foliar", "version": "20261015-abcdef12", "sha256": "d" * 64},
        "result": {"category": "NORMAL", "label": "Healthy", "confidence": 0.93,
                   "top": [[r, p] for r, p in zip(rotulos, probs(), strict=True)]},
        "tiles": {"tile_mm": 150, "distance_mm": 270.0, "labels": rotulos,
                  "boxes": [[rng.randint(0, 4000), rng.randint(0, 3000), 300, 300]
                            for _ in range(recortes)],
                  "heights_mm": [round(rng.uniform(200, 1800), 1) for _ in range(recortes)],
                  "zones": [["meio", "dentro"] for _ in range(recortes)],
                  "probabilities": [probs() for _ in range(recortes)]},
        "retention": {"tier": "REDUZIDA", "priority": 5}, "files": [],
    }
    if modo == "top":
        probs_ = corpo["tiles"].pop("probabilities")
        corpo["tiles"]["top"] = [[rotulos[p.index(max(p))], max(p)] for p in probs_]
    return corpo


def tamanho_do_resultado(recortes: int = 70, classes: int = 6, seed: int = 0,
                         modo: str = "full") -> int:
    """Bytes do JSON canônico de um resultado típico."""
    corpo = corpo_do_resultado(recortes, classes, seed, modo)
    return len(json.dumps(corpo, separators=(",", ":"), sort_keys=True).encode())


def razao_gzip(lote: int = 200, modo: str = "full") -> float:
    """Quanto um lote de ``/edge/records`` encolheria com gzip (nível 6). Medido no corpo
    sintético: probabilidades sorteadas comprimem **pior** que as reais (que se repetem)."""
    import gzip

    corpo = json.dumps({"records": [corpo_do_resultado(seed=i, modo=modo) for i in range(lote)]},
                       separators=(",", ":")).encode()
    return len(gzip.compress(corpo, 6)) / len(corpo)


def simular(cenario: Cenario, h: Hipoteses, cfg: DataBudgetConfig | None, *,
            seed: int = 1, razao_registros: float = 1.0) -> Balanco:
    """Um dia. ``cfg=None`` = a política de antes (sem teto, sem tempestade, sem dedup)."""
    rng = random.Random(seed)
    ret = RetentionConfig()
    regra_doenca = ret.rules["POSSIVEL_DOENCA"]
    pre, pos = 3, 3  # buffer padrão
    b = Balanco()
    fila = {int(p): 0 for p in Priority}  # bytes esperando, por prioridade
    fila_registros = 0  # resultados (metadados): sobem até o teto inteiro
    nunca_cortam = 0
    tempestade = DetectorDeTempestade.de(cfg) if cfg else None
    dedup = DeduplicadorPorPlanta(cfg.k_por_planta, cfg.espacamento_planta_m) if cfg else None
    usado = 0
    banda_por_s = h.banda_subida_bps / 8
    segundos = int(h.horas * 3600)
    infectadas: dict[tuple, bool] = {}
    n_cam = 2 * h.cameras_por_lado
    reqs_registros = 0.0

    def manda(n: int, chave: str) -> None:
        nonlocal usado
        b.soma(b.subiu, chave, n)
        usado += n

    for t in range(segundos):
        x = t * h.velocidade_ms
        eventos_no_segundo = 0
        for cam in range(n_cam):
            lado = "left" if cam < h.cameras_por_lado else "right"
            rua = "L1" if lado == "left" else "L2"
            planta = (rua, round(x / h.espacamento_planta_m))
            if planta not in infectadas:
                infectadas[planta] = rng.random() < cenario.infectadas
            if cenario.tempestade:
                evento = True
            elif infectadas[planta]:
                evento = rng.random() < cenario.deteccao
            else:
                evento = rng.random() < cenario.alarme_falso
            eventos_no_segundo += evento
            # registro do resultado (sempre) — metadados
            b.soma(b.gerado, "registros", h.registro_resultado_bytes)
            fila_registros += h.registro_resultado_bytes * razao_registros
            reqs_registros += 1 / h.registros_por_requisicao
            if tempestade is not None and tempestade.observa(evento, float(t)) == "inicio":
                nunca_cortam += h.registro_evento_bytes  # EVENT_STORM
            if evento:
                b.soma(b.gerado, "eventos", h.registro_evento_bytes)
                nunca_cortam += h.registro_evento_bytes
                arquivos = h.original_bytes + (pre + pos) * h.contexto_bytes
                b.soma(b.gerado, "originais_evento", h.original_bytes)
                b.soma(b.gerado, "contexto", (pre + pos) * h.contexto_bytes)
                motivo = None
                if cfg is not None:
                    sobe, rebaixados = dedup.decide(planta, rng.random(), (str(t),))
                    if not sobe:
                        motivo = "dedup_planta"
                    elif not tempestade.permite_original(float(t)):
                        motivo = "tempestade"
                    for _ in rebaixados:  # o rebaixado ainda não subiu: sai da fila
                        devolve = min(fila[int(regra_doenca.priority)], arquivos)
                        fila[int(regra_doenca.priority)] -= devolve
                        b.retido += devolve
                        b.soma(b.retido_itens, "dedup_planta", 1)
                    if motivo is None and tempestade.ativa:
                        b.retido += (pre + pos) * h.contexto_bytes
                        b.soma(b.retido_itens, "tempestade_contexto", pre + pos)
                        arquivos = h.original_bytes
                if motivo:
                    b.retido += arquivos
                    b.soma(b.retido_itens, motivo, 1)
                else:
                    fila[int(regra_doenca.priority)] += arquivos
            elif rng.random() < h.baixa_confianca:
                b.soma(b.gerado, "originais_baixa_confianca", h.original_bytes)
                fila[int(Priority.ANOMALIA)] += h.original_bytes
            else:
                b.soma(b.gerado, "reduzidas", h.reduzida_bytes)
                fila[int(Priority.NORMAL_COMPACTADA)] += h.reduzida_bytes
            if rng.random() < h.fracao_para_validacao:
                extra = int(h.vistas_extras_por_validacao * h.original_bytes)
                b.soma(b.gerado, "vistas_validacao", extra)
                fila[int(Priority.ANOMALIA)] += extra
        if t % int(h.telemetria_intervalo_s) == 0:
            b.soma(b.gerado, "telemetria", h.telemetria_bytes_por_amostra)
            nunca_cortam += h.telemetria_bytes_por_amostra + h.sobrecarga_requisicao_bytes

        # a fila esvazia nesta ordem, limitada pela banda; o teto corta por prioridade
        banda = banda_por_s
        n = min(nunca_cortam, banda)
        manda(int(n), "nunca_cortam")
        nunca_cortam -= n
        banda -= n
        teto = cfg.bytes_dia if cfg else 0
        if fila_registros and banda > 0 and (not teto or usado < teto):
            n = min(fila_registros, banda)
            manda(int(n), "registros")
            fila_registros -= n
            banda -= n
        for p in sorted(fila):
            if banda <= 0:
                break
            if teto and usado >= cfg.fracao(p) * teto:
                continue
            n = min(fila[p], banda)
            if n <= 0:
                continue
            sobre = n / h.chunk_bytes * h.sobrecarga_requisicao_bytes
            manda(int(n + sobre), f"p{p}")
            fila[p] -= n
            banda -= n
    sobre_registros = int(reqs_registros * h.sobrecarga_requisicao_bytes)
    manda(sobre_registros, "registros")
    b.esperando = int(sum(fila.values()) + fila_registros + nunca_cortam)
    return b


def tabela(h: Hipoteses | None = None, cfg: DataBudgetConfig | None = None) -> list[dict]:
    h = h or Hipoteses()
    cfg = cfg or DataBudgetConfig()
    linhas = []
    for c in CENARIOS:
        antes = simular(c, h, None)
        depois = simular(c, h, cfg)
        gz = simular(c, h, cfg, razao_registros=razao_gzip())
        top = simular(c, h, cfg, razao_registros=razao_gzip(modo="top")
                      * tamanho_do_resultado(modo="top") / tamanho_do_resultado())
        linhas.append({
            "cenario": c.nome, "gerado_gb": antes.total_gerado / 1e9,
            "antes_subiu_gb": antes.total_subiu / 1e9,
            "antes_esperando_gb": antes.esperando / 1e9,
            "depois_subiu_gb": depois.total_subiu / 1e9,
            "depois_esperando_gb": depois.esperando / 1e9,
            "depois_retido_gb": depois.retido / 1e9,
            "depois_retido_itens": depois.retido_itens,
            "gzip_subiu_gb": gz.total_subiu / 1e9,
            "gzip_registros_subiu_gb": gz.subiu.get("registros", 0) / 1e9,
            "gzip_registros_esperando_gb": gz.esperando / 1e9,
            "top_registros_subiu_gb": top.subiu.get("registros", 0) / 1e9,
            "top_subiu_gb": top.total_subiu / 1e9,
            "depois_registros_subiu_gb": depois.subiu.get("registros", 0) / 1e9,
            "depois_subiu_por_tipo_gb": {k: round(v / 1e9, 3) for k, v in depois.subiu.items()},
            "gerado_por_tipo_gb": {k: round(v / 1e9, 3) for k, v in antes.gerado.items()},
        })
    return linhas


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SIMULAÇÃO do volume diário pela Starlink.")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--bytes-dia", type=int, default=None)
    a = ap.parse_args(argv)
    cfg = DataBudgetConfig() if a.bytes_dia is None else DataBudgetConfig(bytes_dia=a.bytes_dia)
    linhas = tabela(cfg=cfg)
    if a.json:
        print(json.dumps({"hipoteses": asdict(Hipoteses()), "teto_dia": cfg.bytes_dia,
                          "resultado_do_corpo_bytes": tamanho_do_resultado(),
                          "linhas": linhas}, ensure_ascii=False, indent=1))
        return 0
    print(f"SIMULADO — teto {cfg.bytes_dia / 1e9:.1f} GB/dia; resultado com 70 recortes = "
          f"{tamanho_do_resultado()} B (full), {tamanho_do_resultado(modo='top')} B (top); "
          f"gzip do lote: {razao_gzip():.2f} (full), {razao_gzip(modo='top'):.2f} (top)")
    print("| cenário | gerado (GB) | antes: subiu | antes: na fila | depois: subiu "
          "(metadados) | depois: na fila | depois: retido | + gzip: subiu (metadados) | "
          "+ gzip e recorte só com a líder: subiu (metadados) |")
    for x in linhas:
        print(f"| {x['cenario']} | {x['gerado_gb']:.1f} | {x['antes_subiu_gb']:.1f} | "
              f"{x['antes_esperando_gb']:.1f} | {x['depois_subiu_gb']:.2f} "
              f"({x['depois_registros_subiu_gb']:.2f}) | {x['depois_esperando_gb']:.1f} | "
              f"{x['depois_retido_gb']:.1f} | {x['gzip_subiu_gb']:.2f} "
              f"({x['gzip_registros_subiu_gb']:.2f}) | {x['top_subiu_gb']:.2f} "
              f"({x['top_registros_subiu_gb']:.2f}) |")
    for x in linhas:
        print(x["cenario"], "gerado:", x["gerado_por_tipo_gb"], "retidos:",
              x["depois_retido_itens"], "subiu depois:", x["depois_subiu_por_tipo_gb"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
