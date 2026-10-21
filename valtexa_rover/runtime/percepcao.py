"""Percepção ativa: quando o quadro fica na dúvida, o Rover olha de novo.

Até aqui cada câmera tirava uma foto por passo e a Edge AI decidia com ela, fosse qual
fosse a qualidade da foto. Um reflexo na folha, a exposição puxada pelo céu, a planta fora
da faixa de profundidade: o quadro cai perto do limiar e a decisão é cara ou coroa. Aqui o
Rover reconhece essa dúvida e age sobre ela, dentro de um orçamento.

O ciclo, por quadro (:meth:`ActivePerception.run`):

1. **A dúvida** (:func:`assess`). Motivos:

   * ``evidencia_na_fronteira`` — o escore de evidência está a menos de uma margem do
     limiar de evento (a margem padrão é a distância entre quase-evento e evento que o
     ``fit_edge`` mediu: no café, 0,036 na escala da LLR média);
   * ``quase_evento`` — a regra por contagem marcou baixa confiança (recorte acima do
     quase-evento);
   * ``anomalia`` — o modelo não reconheceu o que viu;
   * ``profundidade_insuficiente`` — o mapa chegou, mas cobre pouco do quadro: a escala
     dos recortes caiu na distância padrão, que pode estar errada.

   A **região** que decidiu: os recortes a favor da doença líder (LLR > 0, ou acima do
   quase-evento), com ``margem_regiao`` recortes de folga.

2. **Olhar de novo** (até ``max_vistas_extras``). Numa câmera que cumpre
   :class:`~valtexa_rover.hal.interfaces.ActiveCamera`, cada vista extra mede a exposição
   (e o foco, se houver) **naquela região** (estratégia ``regiao``); nas outras, a captura
   se repete (``repeticao``). Entre uma vista e outra o Rover anda alguns centímetros: a
   folha é vista de outro ângulo, e o reflexo de uma vista não está na outra. Com
   profundidade insuficiente, a vista é do quadro inteiro: se ela vier com profundidade
   boa, **substitui** a principal (a escala dos recortes passa a ser a medida).

3. **Alinhar** (:func:`register`). A correlação de fase entre as fotos reduzidas (256 px,
   decodificação parcial do JPEG) dá o deslocamento entre elas. Com pico acima de
   ``min_registro`` e um deslocamento que cabe no movimento do Rover (quase todo horizontal:
   :func:`plausible_shift`), cada recorte da vista extra cai sobre o recorte da principal
   mais perto (até meio recorte de distância). A vista extra só roda o modelo nos recortes da região
   deslocada — no café, ~10 recortes em vez de ~70 (131 ms cada na CPU).

4. **Fundir** (:func:`fuse`). As probabilidades de cada recorte da principal viram a média
   das vistas que o cobriram, e a decisão sai pela **mesma regra calibrada** do perfil
   (:meth:`~valtexa_rover.edge_ai.engine.EdgeAIEngine.decide_tiles`), com o mesmo número de
   recortes: a escala do escore não muda, nenhuma evidência é contada duas vezes. Média, e
   não soma, porque vistas da mesma folha não são independentes. Para quando a dúvida se
   resolve.

5. **O que continua incerto vai para a validação**: o quadro sobe em ORIGINAL, com as vistas
   (``vista1``, ``vista2``), e ``para_validacao`` — é exatamente o exemplo que o agrônomo
   precisa ver e que mais ensina ao aprendizado contínuo (``ia/training/adapt_campo.py``).

Os recortes que sobem para a fusão por planta da nuvem (``tiles``) continuam os da vista
principal, crus: a fusão da nuvem foi calibrada com uma vista por quadro. O que as vistas
extras disseram vai no bloco ``percepcao`` do resultado.

**Orçamento**: ``max_extras_por_minuto`` por câmera e ``tempo_max_ms`` de inferência extra
estimada por quadro (latência por recorte vezes recortes da região). Fora do orçamento, o quadro
segue com a decisão de uma vista e o bloco diz ``skipped``.

**Prioridade da missão** (Nível 4, :mod:`valtexa_rover.runtime.prioridades`): quando a missão
traz o mapa, o trecho de prioridade abaixo de ``prioridade_minima`` não gasta vista extra em
dúvida comum (``skipped: prioridade_baixa``; anomalia continua sendo olhada) e o de prioridade
abaixo de ``prioridade_alta`` deixa ``reserva_prioridade_alta`` vistas por minuto para os
trechos que valem mais (``skipped: orcamento_reservado``). Só o **quanto** insistir muda; a
decisão e a evidência que sobe são as mesmas.

**Não testado em campo**: o alinhamento e a fusão têm testes com cenas sintéticas
(``tests/test_percepcao.py``); a captura dirigida da OAK-D, só com SDK falso.
"""

from __future__ import annotations

import contextlib
import io
import math
import time
from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np

from valtexa_rover.clock import Clock
from valtexa_rover.config import ActivePerceptionConfig, Category
from valtexa_rover.edge_ai.engine import EdgeAIEngine, Inference
from valtexa_rover.edge_ai.vendor.preprocess import InvalidImageError
from valtexa_rover.hal.interfaces import Frame, HardwareUnavailableError

REGISTRO_PX = 256
"""Lado maior das fotos reduzidas para o alinhamento."""

MOTIVOS_DE_DECISAO = ("evidencia_na_fronteira", "quase_evento", "anomalia", "conformal_ambiguo",
                      "fora_da_distribuicao")
"""Dúvida do modelo: o que continuar incerto vai para o agrônomo. ``profundidade_insuficiente``
e ``qualidade_baixa`` são do sensor: pedem outra foto, não um agrônomo."""


MOTIVOS_DE_INFORMACAO = ("fora_da_distribuicao", "qualidade_baixa", "profundidade_insuficiente")
"""Faltou informação para decidir (o modelo não conhece, a foto não serve, a escala não vale)
— diferente de dois diagnósticos empatados."""

CASOS = ("suficiente", "anomalia_confiante", "ambiguo", "informacao_insuficiente")
"""A inspeção adaptativa orientada por incerteza, caso a caso:

* ``suficiente``: aparentemente sadia e a foto basta — registra e segue;
* ``anomalia_confiante``: achado com confiança e dentro do que o modelo conhece — marca a
  posição e guarda a evidência (original e sequência, pela retenção);
* ``ambiguo``: hipóteses empatadas — sequência extra de coleta (vistas, outro ângulo, luz,
  térmica) e, se não resolver, o agrônomo;
* ``informacao_insuficiente``: fora da distribuição, foto ruim ou sem escala — o mesmo, e a
  confiança do modelo não vale até alguém olhar.
"""


def caso_de_inspecao(inf: Inference, duvida: Uncertainty) -> str:
    if any(m in MOTIVOS_DE_INFORMACAO for m in duvida.reasons):
        return "informacao_insuficiente"
    if duvida.uncertain:
        return "ambiguo"
    return "suficiente" if inf.category is Category.NORMAL else "anomalia_confiante"


def medida_termica(termica, regiao: tuple[float, float, float, float] | None) -> dict | None:
    """Temperatura da região da dúvida (percentil 90) contra o resto do quadro (mediana), °C.

    Folha com lesão ou estresse hídrico pode aquecer (menos transpiração), mas a relação no
    café **não foi medida**: o número vai ao agrônomo marcado não calibrado e não muda a
    decisão. O alinhamento térmico x RGB é aproximado pela fração da foto (mesmo campo de
    visão suposto) — a verificar na bancada.
    """
    quadro = termica.capture()
    if quadro is None:
        return None
    t = np.asarray(quadro.temperatures_c, dtype=np.float64)
    if t.ndim != 2 or not t.size:
        return None
    h, w = t.shape
    x0, y0, x1, y1 = regiao or (0.0, 0.0, 1.0, 1.0)
    mascara = np.zeros_like(t, dtype=bool)
    mascara[int(y0 * h):max(int(y0 * h) + 1, int(np.ceil(y1 * h))),
            int(x0 * w):max(int(x0 * w) + 1, int(np.ceil(x1 * w)))] = True
    dentro = t[mascara & np.isfinite(t)]
    fora = t[~mascara & np.isfinite(t)]
    if not dentro.size:
        return None
    # O foco costuma ocupar só parte da região: o percentil 90 dela contra a mediana do resto
    # (a mediana da região diluiria uma mancha pequena).
    regiao_c = float(np.percentile(dentro, 90))
    resto_c = float(np.median(fora)) if fora.size else None
    return {"regiao_c": round(regiao_c, 2), "resto_c": None if resto_c is None
            else round(resto_c, 2),
            "delta_c": None if resto_c is None else round(regiao_c - resto_c, 2),
            "max_regiao_c": round(float(dentro.max()), 2), "emissividade": quadro.emissivity,
            "calibrado": False,
            "nota": "evidência para o agrônomo; a relação temperatura x doença no café não foi "
                    "medida e não muda a decisão"}


# ---------------------------------------------------------------------------------------------
# A dúvida
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Uncertainty:
    reasons: tuple[str, ...]
    region: tuple[float, float, float, float] | None
    """Fração da foto (``x0, y0, x1, y1``); ``None`` = o quadro inteiro."""
    margin: float | None = None
    problemas: tuple[str, ...] = ()
    """Os problemas de captura mais comuns nos recortes medidos (``blurry``, ``dark``...)."""

    @property
    def uncertain(self) -> bool:
        return bool(self.reasons)

    @property
    def decision_doubt(self) -> bool:
        return any(m in MOTIVOS_DE_DECISAO for m in self.reasons)


def _regiao_dos_recortes(inf: Inference, indices: list[int], image_size: tuple[int, int],
                         margem: float) -> tuple[float, float, float, float] | None:
    if not indices:
        return None
    caixas = np.array([inf.tiles[i].box for i in indices], dtype=np.float64)
    lado = float(np.median(caixas[:, 2] - caixas[:, 0]))
    largura, altura = image_size
    x0 = max(0.0, (caixas[:, 0].min() - margem * lado) / largura)
    y0 = max(0.0, (caixas[:, 1].min() - margem * lado) / altura)
    x1 = min(1.0, (caixas[:, 2].max() + margem * lado) / largura)
    y1 = min(1.0, (caixas[:, 3].max() + margem * lado) / altura)
    return (round(float(x0), 4), round(float(y0), 4), round(float(x1), 4), round(float(y1), 4))


def assess(inf: Inference, frame: Frame, engine: EdgeAIEngine,
           cfg: ActivePerceptionConfig) -> Uncertainty:
    """Os motivos de dúvida do quadro e a região que decidiu.

    Só com perfil de recortes (todos os modelos de produção): sem ele o modelo classifica o
    quadro inteiro — é o de demonstração — e não há região nem regra calibrada para fundir.
    """
    perfil = engine.profile
    motivos: list[str] = []
    margem = None
    indices: list[int] = []
    if perfil is None:
        return Uncertainty((), None)
    if inf.tiles:
        regra = perfil.evidence
        if regra is not None and inf.evidence is not None:
            margem = cfg.margem_evidencia if cfg.margem_evidencia is not None else regra.margin()
            if abs(inf.evidence - regra.event_evidence) <= margem:
                motivos.append("evidencia_na_fronteira")
            if inf.leading in regra.llr_tables:
                coluna = engine.labels.index(inf.leading)
                indices = [i for i, t in enumerate(inf.tiles)
                           if regra.tile_llr(inf.leading, t.probabilities[coluna]) > 0]
        elif inf.low_confidence:
            motivos.append("quase_evento")
        if not indices and perfil.evidence is None:
            sadio = engine.labels.index(perfil.healthy_label)
            limiar = (perfil.near_miss_threshold if perfil.near_miss_threshold is not None
                      else perfil.event_threshold)
            indices = [i for i, t in enumerate(inf.tiles)
                       if max(p for j, p in enumerate(t.probabilities) if j != sadio) >= limiar]
    if inf.tiles:
        sadio_rotulo = perfil.healthy_label
        ambiguos = [i for i, t in enumerate(inf.tiles) if t.conformal_set
                    and sadio_rotulo in t.conformal_set
                    and any(c != sadio_rotulo for c in t.conformal_set)]
        if cfg.incerto_em_ambiguidade and len(ambiguos) >= cfg.min_recortes_ambiguos:
            motivos.append("conformal_ambiguo")
            if not indices:
                indices = ambiguos
    # Fora do que a cabeça conhece: a confiança não vale, por mais alta que seja.
    pontuados = [i for i, t in enumerate(inf.tiles) if t.novidade is not None]
    if cfg.incerto_em_novidade and pontuados:
        novos = [i for i in pontuados if inf.tiles[i].novidade > cfg.limiar_novidade]
        decide_fora = inf.category is not Category.NORMAL and any(i in novos for i in indices)
        if novos and (decide_fora or len(novos) / len(pontuados) >= cfg.fracao_recortes_novos):
            motivos.append("fora_da_distribuicao")
            if not indices:
                indices = novos
    problemas: tuple[str, ...] = ()
    medidos = [t.quality for t in inf.tiles if t.quality is not None]
    if medidos:
        ruins = [q for q in medidos if q]
        contagem: dict[str, int] = {}
        for q in ruins:
            for p in q:
                contagem[p] = contagem.get(p, 0) + 1
        problemas = tuple(sorted(contagem, key=lambda p: -contagem[p]))
        if (cfg.incerto_em_qualidade and len(medidos) >= 2
                and len(ruins) / len(medidos) >= cfg.fracao_qualidade_baixa):
            motivos.append("qualidade_baixa")
    if inf.category is Category.ANOMALIA and cfg.incerto_em_anomalia:
        motivos.append("anomalia")
        indices = []
    profundidade_ruim = (frame.depth is not None and inf.rgbd is not None
                         and not inf.rgbd.get("sufficient"))
    if profundidade_ruim and cfg.incerto_sem_profundidade:
        motivos.append("profundidade_insuficiente")
        indices = []  # a escala dos recortes pode estar errada: olha o quadro inteiro
    regiao = (_regiao_dos_recortes(inf, indices, (frame.width, frame.height), cfg.margem_regiao)
              if inf.tiles else None)
    return Uncertainty(tuple(motivos), regiao, margem, problemas)


def qualidade_ruim(inf: Inference, cfg: ActivePerceptionConfig) -> bool:
    """A vista tem fração de recortes com problema de captura acima do limite?"""
    medidos = [t.quality for t in inf.tiles if t.quality is not None]
    return bool(medidos) and sum(1 for q in medidos if q) / len(medidos) >= (
        cfg.fracao_qualidade_baixa)


def pode_reduzir(estado: dict) -> bool:
    """A navegação diz que pode desacelerar: modo autônomo, sem teleoperação, sem obstáculo,
    fora de travessia de zona morta, e ``reducao_permitida``. Sem dado, não pode."""
    return (estado.get("modo") == "autonomo" and not estado.get("teleop")
            and not estado.get("obstaculo") and not estado.get("travessia")
            and estado.get("reducao_permitida") is True)


def deslocamento_na_foto(velocidade_ms: float, latencia_s: float, distancia_mm: float,
                         hfov_deg: float) -> float:
    """Quanto a cena anda na foto (fração da largura) entre a foto e a vista extra."""
    largura_m = 2.0 * (distancia_mm / 1000.0) * math.tan(math.radians(hfov_deg) / 2.0)
    return velocidade_ms * latencia_s / max(largura_m, 1e-6)


def reobservavel(regiao: tuple[float, float, float, float] | None, deslocamento: float) -> bool:
    """A região ainda estará no quadro? Sem saber para que lado a cena anda (depende do lado
    da câmera e do sentido na rua), o conservador: tem de caber deslocada para os dois."""
    if regiao is None:
        return deslocamento < 0.5  # o quadro inteiro: basta sobrar metade em comum
    x0, _y0, x1, _y1 = regiao
    return x0 - deslocamento >= 0.0 and x1 + deslocamento <= 1.0


# ---------------------------------------------------------------------------------------------
# Alinhamento entre vistas
# ---------------------------------------------------------------------------------------------


def gray_small(data: bytes, lado: int = REGISTRO_PX) -> tuple[np.ndarray, float]:
    """A foto em cinza, com lado maior ``lado``, e a escala (px da foto por px reduzido).

    Usa a decodificação parcial do JPEG (``draft``): reduzir 12 MP para 256 px custa poucos
    milissegundos em vez de decodificar a foto inteira.
    """
    from PIL import Image

    with Image.open(io.BytesIO(data)) as img:
        largura = img.size[0]
        img.draft("L", (max(1, img.size[0] // 8), max(1, img.size[1] // 8)))
        cinza = img.convert("L")
        cinza.thumbnail((lado, lado))
        return np.asarray(cinza, dtype=np.float64), largura / float(cinza.size[0])


def register(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Correlação de fase: ``(dx, dy, pico)`` com ``ponto_em_a = ponto_em_b + (dx, dy)``.

    ``pico`` é o máximo da correlação normalizada (0 a 1): perto de 1 para a mesma cena
    deslocada, perto de ``1/N`` para cenas sem relação. Uma janela de Hann tira o efeito das
    bordas; o deslocamento sai com resolução de meio pixel reduzido (interpolação parabólica
    em volta do pico).
    """
    if a.shape != b.shape:
        altura = min(a.shape[0], b.shape[0])
        largura = min(a.shape[1], b.shape[1])
        a, b = a[:altura, :largura], b[:altura, :largura]
    janela = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    fa = np.fft.fft2((a - a.mean()) * janela)
    fb = np.fft.fft2((b - b.mean()) * janela)
    cruzado = fa * np.conj(fb)
    cruzado /= np.abs(cruzado) + 1e-12
    corr = np.fft.ifft2(cruzado).real
    iy, ix = np.unravel_index(int(np.argmax(corr)), corr.shape)
    pico = float(corr[iy, ix])

    def refina(c_menos: float, c0: float, c_mais: float) -> float:
        den = c_menos - 2 * c0 + c_mais
        return 0.0 if abs(den) < 1e-12 else float(np.clip(0.5 * (c_menos - c_mais) / den,
                                                          -0.5, 0.5))

    h, w = corr.shape
    dy = iy + refina(corr[(iy - 1) % h, ix], corr[iy, ix], corr[(iy + 1) % h, ix])
    dx = ix + refina(corr[iy, (ix - 1) % w], corr[iy, ix], corr[iy, (ix + 1) % w])
    if dy > h / 2:
        dy -= h
    if dx > w / 2:
        dx -= w
    return float(dx), float(dy), pico


def plausible_shift(dx_px: float, dy_px: float, largura: int, altura: int,
                    cfg: ActivePerceptionConfig) -> bool:
    """O deslocamento cabe no movimento do Rover entre duas fotos?

    A câmera é fixa no mastro e o Rover anda ao longo da linha: entre a foto principal e a
    extra, a cena corre na horizontal e quase nada na vertical. Um pico da correlação de fase
    fora disso não é a cena andando — é um reflexo de alto contraste, que existe numa foto e
    não na outra, puxando o alinhamento. Aí a vista não casa recorte a recorte.
    """
    return (abs(dy_px) <= cfg.max_desvio_vertical * altura
            and abs(dx_px) <= cfg.max_desvio_horizontal * largura)


def _desloca_regiao(regiao, dx_px: float, dy_px: float, largura: int, altura: int):
    """A região da principal, nas coordenadas da vista extra (``b = a - d``)."""
    if regiao is None:
        return None
    x0, y0, x1, y1 = regiao
    ddx, ddy = dx_px / largura, dy_px / altura
    caixa = (max(0.0, x0 - ddx), max(0.0, y0 - ddy), min(1.0, x1 - ddx), min(1.0, y1 - ddy))
    if caixa[2] <= caixa[0] or caixa[3] <= caixa[1]:
        return None  # a região saiu da foto: a vista inteira
    return tuple(round(v, 4) for v in caixa)


# ---------------------------------------------------------------------------------------------
# Fusão
# ---------------------------------------------------------------------------------------------


@dataclass(slots=True)
class View:
    frame: Frame
    inference: Inference
    strategy: str
    shift_px: tuple[float, float] = (0.0, 0.0)
    peak: float = 0.0
    aligned: bool = False
    region: tuple[float, float, float, float] | None = None


@dataclass(slots=True)
class Fusion:
    inference: Inference
    indices: list[int] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)
    probabilities: list[list[float]] = field(default_factory=list)


def fuse(engine: EdgeAIEngine, primary: Inference, views: list[View]) -> Fusion:
    """A decisão com as vistas: média das probabilidades por recorte, mesma regra do perfil."""
    vistas = [v for v in views if v.aligned]
    if not primary.tiles:
        return Fusion(primary)
    probs = np.array([t.probabilities for t in primary.tiles], dtype=np.float64)
    soma = probs.copy()
    contagem = np.ones(len(probs), dtype=np.int64)
    centros = np.array([((t.box[0] + t.box[2]) / 2.0, (t.box[1] + t.box[3]) / 2.0)
                        for t in primary.tiles])
    lado = float(np.median([t.box[2] - t.box[0] for t in primary.tiles]))
    for v in vistas:
        if not v.inference.tiles:
            continue
        outros = np.array([((t.box[0] + t.box[2]) / 2.0 + v.shift_px[0],
                            (t.box[1] + t.box[3]) / 2.0 + v.shift_px[1])
                           for t in v.inference.tiles])
        outras_probs = np.array([t.probabilities for t in v.inference.tiles], dtype=np.float64)
        distancias = np.linalg.norm(centros[:, None, :] - outros[None, :, :], axis=2)
        mais_perto = distancias.argmin(axis=1)
        casou = distancias[np.arange(len(centros)), mais_perto] <= 0.5 * lado
        soma[casou] += outras_probs[mais_perto[casou]]
        contagem[casou] += 1
    if int(contagem.max()) == 1:
        return Fusion(primary)
    fundidas = soma / contagem[:, None]
    d = engine.decide_tiles(fundidas)
    n_vistas = 1 + len(vistas)
    nova = replace(primary, label=d.label, confidence=round(d.confidence, 6),
                   category=d.category, low_confidence=d.low_confidence, top=d.top,
                   reason=f"{d.reason} (fusão de {n_vistas} vistas)".strip(),
                   event_tiles=d.event_tiles, leading=d.leading,
                   evidence=None if d.evidence is None else round(d.evidence, 6))
    indices = [int(i) for i in np.flatnonzero(contagem > 1)]
    return Fusion(nova, indices, [int(contagem[i]) for i in indices],
                  [[round(float(p), 5) for p in fundidas[i]] for i in indices])


# ---------------------------------------------------------------------------------------------
# O ciclo
# ---------------------------------------------------------------------------------------------


@dataclass(slots=True)
class Outcome:
    """O que a percepção ativa devolve ao Runtime para gravar o quadro."""

    frame: Frame
    inference: Inference
    block: dict | None = None
    """O bloco ``percepcao`` do resultado (``None``: o quadro não estava em dúvida)."""
    extra_frames: list[Frame] = field(default_factory=list)
    """Vistas extras cujo original deve ser guardado (para a validação)."""
    for_validation: bool = False


def _resumo(inf: Inference) -> dict:
    return {"category": inf.category.value, "label": inf.label, "evidence": inf.evidence,
            "low_confidence": inf.low_confidence, "event_tiles": inf.event_tiles}


class ActivePerception:
    """Orçamento, vistas extras, alinhamento e fusão — por câmera, dentro da missão."""

    def __init__(self, cfg: ActivePerceptionConfig, clock: Clock) -> None:
        self.cfg = cfg
        self.clock = clock
        self._uso: dict[str, deque[float]] = {}
        self.stats = {"avaliados": 0, "incertos": 0, "vistas": 0, "resolvidos": 0,
                      "para_validacao": 0, "sem_orcamento": 0, "falha_captura": 0,
                      "sem_alinhamento": 0, "vista_promovida": 0, "prioridade_baixa": 0,
                      "orcamento_reservado": 0}

    def _disponivel(self, camera_id: str) -> int:
        agora = self.clock.monotonic()
        uso = self._uso.setdefault(camera_id, deque())
        while uso and agora - uso[0] >= 60.0:
            uso.popleft()
        return self.cfg.max_extras_por_minuto - len(uso)

    def _consome(self, camera_id: str) -> None:
        self._uso.setdefault(camera_id, deque()).append(self.clock.monotonic())

    @staticmethod
    def _custo_estimado(inf: Inference, regiao, frame: Frame) -> float:
        """Milissegundos de inferência de uma vista extra: latência por recorte vezes recortes."""
        if not inf.tiles:
            return inf.latency_ms
        por_recorte = inf.latency_ms / len(inf.tiles)
        if regiao is None:
            return inf.latency_ms
        x0, y0, x1, y1 = regiao
        dentro = sum(1 for t in inf.tiles
                     if x0 <= (t.box[0] + t.box[2]) / 2 / frame.width <= x1
                     and y0 <= (t.box[1] + t.box[3]) / 2 / frame.height <= y1)
        return por_recorte * max(1, dentro)

    def run(self, engine: EdgeAIEngine, camera, frame: Frame, inf: Inference,
            controller=None, prioridade: float | None = None, luz=None,
            termica=None) -> Outcome:
        cfg = self.cfg
        if not cfg.ativa or cfg.max_vistas_extras == 0 or camera is None:
            return Outcome(frame, inf)
        self.stats["avaliados"] += 1
        duvida = assess(inf, frame, engine, cfg)
        caso = caso_de_inspecao(inf, duvida)
        self.stats[f"caso_{caso}"] = self.stats.get(f"caso_{caso}", 0) + 1
        if not duvida.uncertain:
            return Outcome(frame, inf)
        self.stats["incertos"] += 1
        bloco: dict = {"caso": caso, "reasons": list(duvida.reasons), "region": duvida.region,
                       "margin": duvida.margin, "before": _resumo(inf), "views": [],
                       "acted": False, "skipped": None, "resolved": False,
                       "promoted_view": False, "prioridade": prioridade}
        # O piso do orçamento: o trecho que vale pouco não gasta a reserva dos que valem mais
        # (vale a cada vista, não só na primeira).
        piso = 0
        if prioridade is not None and "anomalia" not in duvida.reasons:
            if prioridade < cfg.prioridade_minima:
                bloco["skipped"] = "prioridade_baixa"
                self.stats["prioridade_baixa"] += 1
                return self._fecha(Outcome(frame, inf, bloco), engine)
            if prioridade < cfg.prioridade_alta:
                piso = cfg.reserva_prioridade_alta
        if piso and self._disponivel(frame.camera_id) <= piso:
            bloco["skipped"] = "orcamento_reservado"
            self.stats["orcamento_reservado"] += 1
            return self._fecha(Outcome(frame, inf, bloco), engine)
        custo = self._custo_estimado(inf, duvida.region, frame)
        if self._disponivel(frame.camera_id) <= 0:
            bloco["skipped"] = "orcamento"
        elif custo > cfg.tempo_max_ms:
            bloco["skipped"] = "tempo"
        if bloco["skipped"]:
            self.stats["sem_orcamento"] += 1
            return self._fecha(Outcome(frame, inf, bloco), engine)

        # A região ainda estará no quadro quando a vista extra sair? Pela velocidade da
        # navegação; se não, desacelera (quando a navegação deixa) ou não insiste.
        estado = {}
        if controller is not None:
            try:
                estado = controller.state() or {}
            except Exception:  # a navegação é de outro processo: segue sem saber
                estado = {}
        velocidade = estado.get("speed_ms")
        distancia = frame.distance_mm or inf.distance_mm
        hfov = inf.hfov_deg or frame.hfov_deg
        if velocidade is not None and distancia and hfov:
            desloc = deslocamento_na_foto(float(velocidade), cfg.latencia_vista_s,
                                          float(distancia), float(hfov))
            bloco["deslocamento_previsto"] = round(desloc, 4)
            if (cfg.reduzir_velocidade and pode_reduzir(estado)
                    and hasattr(controller, "pedir_velocidade")):
                try:
                    controller.pedir_velocidade(cfg.fator_velocidade, cfg.duracao_reducao_s,
                                                f"percepção ativa: {frame.camera_id}")
                    bloco["velocidade"] = {"fator": cfg.fator_velocidade,
                                           "duracao_s": cfg.duracao_reducao_s}
                    self.stats["desaceleracoes"] = self.stats.get("desaceleracoes", 0) + 1
                    desloc *= cfg.fator_velocidade
                except Exception:  # pedido não chegou: decide com a velocidade de agora
                    bloco["velocidade"] = None
            elif cfg.reduzir_velocidade:
                bloco["velocidade"] = {"negada": "a navegação não disse que pode reduzir"}
            if not reobservavel(duvida.region, desloc):
                bloco["skipped"] = "fora_do_campo"
                self.stats["fora_do_campo"] = self.stats.get("fora_do_campo", 0) + 1
                return self._fecha(Outcome(frame, inf, bloco), engine)

        if cfg.pedir_pausa_s > 0 and controller is not None and hasattr(controller,
                                                                        "pedir_pausa"):
            try:
                controller.pedir_pausa(cfg.pedir_pausa_s, f"percepção ativa: {frame.camera_id}")
                bloco["pause_requested_s"] = cfg.pedir_pausa_s
            except Exception:  # a navegação é de outro processo: não perde o quadro
                bloco["pause_requested_s"] = None

        # Medida complementar: a térmica, quando o hardware tem. Só evidência.
        if termica is not None and cfg.usar_termica:
            try:
                bloco["termica"] = medida_termica(termica, duvida.region)
                self.stats["termica"] = self.stats.get("termica", 0) + 1
            except Exception:  # driver da térmica fora: segue sem ela
                bloco["termica"] = None
        # Luz auxiliar pelo problema medido: escuro pede mais, estourado pede menos. Volta
        # ao nível de antes depois das vistas (o próximo quadro não é desta dúvida).
        nivel_antes = None
        if luz is not None and cfg.ajustar_luz and duvida.problemas:
            try:
                nivel_antes = luz.level()
                if "dark" in duvida.problemas:
                    bloco["luz"] = luz.set_level(nivel_antes + cfg.passo_luz)
                elif "overexposed" in duvida.problemas:
                    bloco["luz"] = luz.set_level(nivel_antes - cfg.passo_luz)
                else:
                    nivel_antes = None
            except Exception:  # driver da luz fora: segue sem ela
                nivel_antes = None
        try:
            return self._vistas(engine, camera, frame, inf, duvida, bloco, piso)
        finally:
            if nivel_antes is not None:
                with contextlib.suppress(Exception):
                    luz.set_level(nivel_antes)

    def _vistas(self, engine: EdgeAIEngine, camera, frame: Frame, inf: Inference,
                duvida: Uncertainty, bloco: dict, piso: int) -> Outcome:
        cfg = self.cfg
        principal, inf_principal = frame, inf
        base_cinza, base_escala = gray_small(frame.data)
        vistas: list[View] = []
        fusao = Fusion(inf)
        for _ in range(cfg.max_vistas_extras):
            if self._disponivel(frame.camera_id) <= piso:
                bloco["skipped"] = "orcamento_reservado" if piso else "orcamento"
                self.stats["orcamento_reservado" if piso else "sem_orcamento"] += 1
                break
            self._consome(frame.camera_id)
            regiao_pedida = None if "profundidade_insuficiente" in duvida.reasons else (
                duvida.region)
            dirigida = hasattr(camera, "capture_region")
            estrategia = "regiao" if dirigida else "repeticao"
            inicio = time.perf_counter()
            try:
                if dirigida:
                    extra = camera.capture_region(regiao_pedida or (0.0, 0.0, 1.0, 1.0),
                                                  principal.distance_mm)
                else:
                    extra = camera.capture()
            except HardwareUnavailableError as exc:
                self.stats["falha_captura"] += 1
                bloco["views"].append({"strategy": estrategia, "error": str(exc)})
                break
            extra = replace(extra, view=estrategia)
            self.stats["vistas"] += 1
            bloco["acted"] = True
            try:
                cinza, escala = gray_small(extra.data)
                dx, dy, pico = register(base_cinza, cinza)
                dx_px, dy_px = dx * base_escala, dy * base_escala
                alinhada = (pico >= cfg.min_registro
                            and abs(base_escala - escala) < 0.02 * escala
                            and plausible_shift(dx_px, dy_px, principal.width, principal.height,
                                                cfg))
                if not alinhada:
                    self.stats["sem_alinhamento"] += 1
                    if regiao_pedida is not None:
                        # Sem alinhamento a vista não casa recorte a recorte e a fusão não a
                        # usaria: não roda o modelo no quadro inteiro à toa.
                        bloco["views"].append({
                            "strategy": estrategia, "captured_at": extra.captured_at.isoformat(),
                            "registration_peak": round(pico, 4), "aligned": False,
                            "tiles": 0})
                        continue
                regiao = (_desloca_regiao(regiao_pedida, dx_px, dy_px, principal.width,
                                          principal.height) if alinhada else None)
                inf_extra = engine.infer(extra.data, extra.distance_mm, extra.camera_height_mm,
                                         extra.hfov_deg, depth=extra.depth, region=regiao)
            except (InvalidImageError, OSError, ValueError) as exc:
                # Vista ilegível: o quadro segue com o que já tem.
                self.stats["falha_captura"] += 1
                bloco["views"].append({"strategy": estrategia, "error": str(exc)})
                break
            if qualidade_ruim(inf_extra, cfg):
                # A vista nova também saiu fora da régua (borrada, escura): não entra na fusão.
                self.stats["vista_ruim"] = self.stats.get("vista_ruim", 0) + 1
                bloco["views"].append({
                    "strategy": estrategia, "captured_at": extra.captured_at.isoformat(),
                    "aligned": alinhada, "quality": "ruim", "tiles": len(inf_extra.tiles)})
                continue
            vista = View(extra, inf_extra, estrategia, (dx_px, dy_px), pico, alinhada, regiao)
            bloco["views"].append({
                "strategy": estrategia, "captured_at": extra.captured_at.isoformat(),
                "shift_px": [round(dx_px, 1), round(dy_px, 1)],
                "registration_peak": round(pico, 4), "aligned": alinhada, "region": regiao,
                "tiles": len(inf_extra.tiles), "category": inf_extra.category.value,
                "label": inf_extra.label, "evidence": inf_extra.evidence,
                "distance_mm": inf_extra.distance_mm,
                "depth_sufficient": (inf_extra.rgbd or {}).get("sufficient"),
                "latency_ms": round((time.perf_counter() - inicio) * 1000, 1),
            })
            # Profundidade ruim na principal e boa nesta vista inteira: esta vira a principal
            # (a escala dos recortes passa a ser a medida).
            if ("profundidade_insuficiente" in duvida.reasons and regiao_pedida is None
                    and (inf_extra.rgbd or {}).get("sufficient") and inf_extra.tiles):
                principal, inf_principal = extra, inf_extra
                base_cinza, base_escala = cinza, escala
                bloco["promoted_view"] = True
                self.stats["vista_promovida"] += 1
                vistas = []
                fusao = Fusion(inf_extra)
                duvida = assess(inf_extra, extra, engine, cfg)
                bloco["reasons_after_promotion"] = list(duvida.reasons)
                if not duvida.decision_doubt:
                    break
                continue
            vistas.append(vista)
            fusao = fuse(engine, inf_principal, vistas)
            depois = assess(fusao.inference, principal, engine, cfg)
            if not depois.decision_doubt and not ("profundidade_insuficiente" in depois.reasons
                                                  and "profundidade_insuficiente"
                                                  in duvida.reasons):
                break
        bloco["after"] = _resumo(fusao.inference)
        if fusao.indices:
            bloco["fused_tiles"] = {"indices": fusao.indices, "views": fusao.counts,
                                    "probabilities": fusao.probabilities}
        guardar = [v.frame for v in vistas]
        return self._fecha(Outcome(principal, fusao.inference, bloco, guardar), engine)

    def _fecha(self, desfecho: Outcome, engine: EdgeAIEngine) -> Outcome:
        """Resolvida ou não; o que fica para a validação; que vistas guardar.

        Só a dúvida **de decisão** vai para a validação. Profundidade ruim sozinha é do
        sensor, não do modelo: se ela mandasse o original de 12 MP para cima, uma OAK-D sem
        medida em campo encheria o disco em uma missão.
        """
        cfg = self.cfg
        assert desfecho.block is not None
        restante = assess(desfecho.inference, desfecho.frame, engine, cfg)
        resolvida = bool(desfecho.block["acted"]) and not restante.uncertain
        desfecho.block["resolved"] = resolvida
        desfecho.block["reasons_left"] = list(restante.reasons)
        desfecho.for_validation = not resolvida and restante.decision_doubt
        desfecho.block["for_validation"] = desfecho.for_validation
        if resolvida:
            self.stats["resolvidos"] += 1
        if desfecho.for_validation:
            self.stats["para_validacao"] += 1
        if cfg.guardar_vistas == "nunca" or (cfg.guardar_vistas == "incertas"
                                             and not desfecho.for_validation):
            desfecho.extra_frames = []
        return desfecho

    def snapshot(self) -> dict:
        return dict(self.stats)
