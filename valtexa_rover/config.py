"""Configuration Manager: tudo que muda o comportamento do Rover sem mudar código.

A configuração é um JSON (``rover.json``) com seções; o que não vier fica no
padrão daqui. Chave desconhecida é **erro** (:class:`ConfigError`), não
silêncio: um ``"retencao"`` digitado errado não pode deixar o robô guardando
tudo em resolução máxima sem ninguém perceber.

As seções:

* ``classificacao`` — limiares de baixa confiança e de anomalia (seção 14);
* ``retencao`` — o que fica de cada categoria: só metadados, imagem reduzida ou
  original, por quantos dias e com que prioridade de upload (seções 16, 17, 19);
* ``buffer`` — quadros antes/depois de um evento, duração, resolução (seção 15);
* ``sync`` — chunk, lote, *backoff* (seção 18);
* ``saude`` — limites do Health Monitor;
* ``profundidade`` — a fusão RGB + profundidade da OAK-D (o que é planta e o que é fundo);
* ``percepcao_ativa`` — vistas extras quando o quadro fica na dúvida;
* ``orcamento_dados`` — o teto de bytes pela Starlink do Rover (dia e missão), a ordem de
  corte, a tempestade de eventos e a deduplicação por planta.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from enum import IntEnum
from pathlib import Path
from typing import Any

from valtexa_rover._compat import StrEnum


class ConfigError(ValueError):
    """Configuração inválida. A mensagem diz a chave e o problema."""


class Category(StrEnum):
    """Categorias da Edge AI (brief, seção 14)."""

    NORMAL = "NORMAL"
    POSSIVEL_PRAGA = "POSSIVEL_PRAGA"
    POSSIVEL_DOENCA = "POSSIVEL_DOENCA"
    ANOMALIA = "ANOMALIA"


class Tier(StrEnum):
    """Os três níveis de dados (brief, seção 16)."""

    METADADOS = "METADADOS"
    REDUZIDA = "REDUZIDA"
    ORIGINAL = "ORIGINAL"


class Priority(IntEnum):
    """Prioridade de upload (brief, seção 19). Menor número sobe primeiro."""

    EVENTO_CRITICO = 1
    DOENCA_PRAGA = 2
    ANOMALIA = 3
    CONCLUSAO_MISSAO = 4
    NORMAL_COMPACTADA = 5
    SECUNDARIO = 6


@dataclass(frozen=True, slots=True)
class ClassificationConfig:
    low_confidence_below: float = 0.60
    """Abaixo disto o resultado é marcado ``baixa_confianca`` (e preserva o original)."""
    anomaly_below: float = 0.30
    """Abaixo disto nenhuma classe foi reconhecida: vira ANOMALIA (fora do que o modelo conhece)."""

    def validate(self) -> None:
        if not 0 <= self.anomaly_below <= self.low_confidence_below <= 1:
            raise ConfigError(
                "classificacao: exige 0 ≤ anomaly_below ≤ low_confidence_below ≤ 1."
            )


@dataclass(frozen=True, slots=True)
class RetentionRule:
    tier: Tier
    retention_days: int
    priority: Priority
    max_side: int = 0
    """Lado maior da imagem reduzida, em pixels (só vale para ``REDUZIDA``)."""
    jpeg_quality: int = 70
    preserve_sequence: bool = False
    """Guardar também a sequência do buffer circular (quadros antes/depois)."""

    def validate(self, nome: str) -> None:
        if self.retention_days < 0:
            raise ConfigError(f"retencao.{nome}: retention_days negativo.")
        if self.tier is Tier.REDUZIDA and self.max_side < 16:
            raise ConfigError(f"retencao.{nome}: REDUZIDA exige max_side ≥ 16.")
        if not 1 <= self.jpeg_quality <= 95:
            raise ConfigError(f"retencao.{nome}: jpeg_quality fora de 1 a 95.")


def _regras_padrao() -> dict[str, RetentionRule]:
    return {
        Category.NORMAL: RetentionRule(Tier.REDUZIDA, 7, Priority.NORMAL_COMPACTADA, 320, 60),
        Category.POSSIVEL_PRAGA: RetentionRule(
            Tier.ORIGINAL, 90, Priority.DOENCA_PRAGA, preserve_sequence=True
        ),
        Category.POSSIVEL_DOENCA: RetentionRule(
            Tier.ORIGINAL, 90, Priority.DOENCA_PRAGA, preserve_sequence=True
        ),
        Category.ANOMALIA: RetentionRule(
            Tier.ORIGINAL, 90, Priority.ANOMALIA, preserve_sequence=True
        ),
        "BAIXA_CONFIANCA": RetentionRule(Tier.ORIGINAL, 30, Priority.ANOMALIA),
    }


@dataclass(frozen=True, slots=True)
class RetentionConfig:
    """Política de retenção. ``BAIXA_CONFIANCA`` vale por cima da categoria."""

    rules: dict[str, RetentionRule] = field(default_factory=_regras_padrao)
    normal_sample_every: int = 0
    """Uma a cada N imagens NORMAL vai em ORIGINAL, para validação (0 = nunca)."""

    def rule_for(self, category: Category, low_confidence: bool) -> RetentionRule:
        if low_confidence and "BAIXA_CONFIANCA" in self.rules:
            regra = self.rules["BAIXA_CONFIANCA"]
            base = self.rules[category]
            # Baixa confiança nunca *rebaixa*: uma DOENÇA duvidosa continua com a
            # prioridade e a sequência de doença.
            return replace(
                regra,
                priority=min(regra.priority, base.priority),
                retention_days=max(regra.retention_days, base.retention_days),
                preserve_sequence=regra.preserve_sequence or base.preserve_sequence,
            )
        return self.rules[category]

    def validate(self) -> None:
        for categoria in Category:
            if categoria not in self.rules:
                raise ConfigError(f"retencao: falta a regra de {categoria.value}.")
        for nome, regra in self.rules.items():
            regra.validate(str(nome))
        if self.normal_sample_every < 0:
            raise ConfigError("retencao.normal_sample_every negativo.")


@dataclass(frozen=True, slots=True)
class BufferConfig:
    pre_frames: int = 3
    post_frames: int = 3
    max_duration_s: float = 10.0
    """Quadro mais velho que isto (em relação ao evento) não entra na sequência."""
    max_side: int = 1024
    """Resolução dos quadros de contexto guardados (o do evento fica original)."""
    jpeg_quality: int = 85
    priority: Priority = Priority.ANOMALIA

    def validate(self) -> None:
        if self.pre_frames < 0 or self.post_frames < 0:
            raise ConfigError("buffer: pre_frames e post_frames não podem ser negativos.")
        if self.max_duration_s <= 0:
            raise ConfigError("buffer.max_duration_s precisa ser positivo.")
        if self.max_side < 16:
            raise ConfigError("buffer.max_side precisa ser ≥ 16.")


@dataclass(frozen=True, slots=True)
class SyncConfig:
    chunk_size: int = 256 * 1024
    batch_size: int = 200
    """Registros (resultados/eventos/telemetria) por requisição."""
    backoff_base_s: float = 1.0
    backoff_max_s: float = 60.0
    max_attempts: int = 0
    """Tentativas antes de desistir de um item por erro do servidor (0 = nunca desiste)."""
    comprimir_registros: bool = False
    """Lote de ``/edge/records`` em gzip (~0,37 do tamanho no corpo sintético com 70
    recortes). Desligado por padrão: só ligar com a plataforma que aceita
    ``Content-Encoding: gzip`` (rodada 4) já no ar."""

    def validate(self) -> None:
        if self.chunk_size < 1024:
            raise ConfigError("sync.chunk_size precisa ser ≥ 1024 bytes.")
        if self.batch_size < 1:
            raise ConfigError("sync.batch_size precisa ser ≥ 1.")
        if self.backoff_base_s <= 0 or self.backoff_max_s < self.backoff_base_s:
            raise ConfigError("sync: exige 0 < backoff_base_s ≤ backoff_max_s.")


@dataclass(frozen=True, slots=True)
class HealthConfig:
    min_free_bytes: int = 2 * 1024**3
    """Abaixo disto: falha ``storage_low`` e a retenção apaga o que já subiu e venceu."""
    max_temperature_c: float = 70.0

    def validate(self) -> None:
        if self.min_free_bytes < 0:
            raise ConfigError("saude.min_free_bytes negativo.")


MODOS_PROFUNDIDADE = ("filtrar", "anotar", "desligado")


@dataclass(frozen=True, slots=True)
class DepthConfig:
    """Fusão RGB + profundidade (:mod:`valtexa_rover.edge_ai.rgbd`).

    A profundidade da OAK-D separa a planta (perto da câmera) do que aparece pelos vãos do
    dossel (a outra rua, o céu, o chão longe). Os padrões são da geometria do café — câmera
    a ~27 cm da saia — e **não foram medidos em campo**: conferir na bancada.
    """

    modo: str = "filtrar"
    """``filtrar``: recorte que a profundidade diz ser fundo não vai ao modelo (nem à
    evidência); ``anotar``: só mede e manda à nuvem; ``desligado``: ignora o mapa."""
    frente_mm: float = 150.0
    """Até quanto à frente da distância da planta ainda é planta (folha solta, ramo)."""
    tras_mm: float = 450.0
    """Até quanto atrás ainda é a mesma planta (o miolo do dossel)."""
    min_validos_recorte: float = 0.25
    """Fração mínima de pixels com medida para julgar um recorte; abaixo, ele fica (não se
    descarta o que não se mediu)."""
    min_na_planta: float = 0.35
    """Fração dos pixels medidos do recorte dentro da faixa da planta para ele ser planta."""
    min_cobertura_quadro: float = 0.15
    """Abaixo desta fração medida no centro do quadro, a profundidade não decide nada (sol
    forte, superfície sem textura) — e o quadro conta como incerto na percepção ativa."""
    usar_distancia: bool = True
    """Usar a distância da planta pelo pico do histograma (mais robusta que a mediana quando
    o fundo aparece pelos vãos) para o tamanho dos recortes."""

    def validate(self) -> None:
        if self.modo not in MODOS_PROFUNDIDADE:
            raise ConfigError(f"profundidade.modo: use {', '.join(MODOS_PROFUNDIDADE)}.")
        if self.frente_mm < 0 or self.tras_mm <= 0:
            raise ConfigError("profundidade: frente_mm ≥ 0 e tras_mm > 0.")
        for nome in ("min_validos_recorte", "min_na_planta", "min_cobertura_quadro"):
            if not 0.0 <= getattr(self, nome) <= 1.0:
                raise ConfigError(f"profundidade.{nome} fora de [0, 1].")


@dataclass(frozen=True, slots=True)
class ActivePerceptionConfig:
    """Percepção ativa (:mod:`valtexa_rover.runtime.percepcao`): na dúvida, olhar de novo.

    Quando o quadro cai na faixa de incerteza, a câmera tira vistas extras da região que
    decidiu (com exposição — e foco, se a câmera tiver — naquela região), as vistas são
    alinhadas e a decisão sai da média dos recortes. O que continua incerto vai inteiro
    para a validação do agrônomo: é o dado que mais ensina ao aprendizado contínuo.
    """

    ativa: bool = True
    max_vistas_extras: int = 2
    """Vistas extras por quadro, no máximo (para antes, se a dúvida se resolver)."""
    max_extras_por_minuto: int = 6
    """Por câmera. O orçamento protege o ritmo da missão."""
    tempo_max_ms: float = 4000.0
    """Inferência extra estimada por quadro acima disto: não insiste."""
    margem_evidencia: float | None = None
    """Meia-largura da faixa de incerteza em volta do limiar de evento. ``None``: a do
    perfil (distância entre quase-evento e evento, medida pelo ``fit_edge``)."""
    incerto_sem_profundidade: bool = True
    """Quadro sem distância medida (ou com cobertura baixa) conta como incerto."""
    incerto_em_anomalia: bool = True
    min_registro: float = 0.06
    """Pico mínimo da correlação de fase para alinhar as vistas recorte a recorte; abaixo,
    a vista entra só pela região inteira."""
    max_desvio_vertical: float = 0.06
    """Deslocamento vertical máximo entre as vistas, em fração da altura da foto: a câmera é
    fixa no mastro e o Rover anda na horizontal (trepidação do terreno fica abaixo disto)."""
    max_desvio_horizontal: float = 0.5
    """Deslocamento horizontal máximo, em fração da largura (acima disto as vistas quase não
    se sobrepõem)."""
    margem_regiao: float = 1.0
    """Quantos recortes de folga em volta da região que decidiu."""
    pedir_pausa_s: float = 0.0
    """Pedir à navegação para parar este tempo durante as vistas extras (0 = não pede; a
    pilha de navegação precisa ler o pedido — ver ``NavegacaoPorArquivo.pedir_pausa``)."""
    guardar_vistas: str = "incertas"
    """``incertas``: originais das vistas só quando a dúvida não se resolveu (vão para a
    validação); ``sempre``; ``nunca``."""
    prioridade_minima: float = 0.15
    """Com as prioridades da missão (Nível 4, :mod:`valtexa_rover.runtime.prioridades`): no
    trecho abaixo disto, dúvida comum não ganha vista extra (anomalia ganha sempre)."""
    reserva_prioridade_alta: int = 2
    """Vistas extras por minuto que o trecho de prioridade abaixo de ``prioridade_alta`` não
    pode gastar: ficam para onde vale mais."""
    prioridade_alta: float = 0.5
    incerto_em_ambiguidade: bool = True
    """Conjunto conformal ambíguo (sadio **e** doença no mesmo recorte) conta como dúvida."""
    min_recortes_ambiguos: int = 2
    incerto_em_qualidade: bool = True
    """Recortes fora da régua da captura do modelo (borrado, escuro, estourado) contam como
    dúvida — de sensor, não de decisão: não mandam original para a validação sozinhos."""
    fracao_qualidade_baixa: float = 0.5
    """Fração dos recortes medidos com problema a partir da qual o quadro é de qualidade baixa
    (e a vista extra com essa fração é descartada da fusão)."""
    reduzir_velocidade: bool = False
    """Pedir à navegação que reduza a velocidade enquanto olha de novo
    (``NavegacaoPorArquivo.pedir_velocidade``). Desligado como a pausa: o protocolo com a
    navegação é a definir, e o pedido só sai com a navegação dizendo que pode."""
    fator_velocidade: float = 0.3
    duracao_reducao_s: float = 1.5
    """**Hipótese de engenharia** (1,5 s): o tempo real sai dos testes de campo."""
    latencia_vista_s: float = 0.4
    """Da foto principal à vista extra (no vídeo de lavoura, 88% das vistas alinharam a 0,4 s;
    79% a 0,8 s). Com a velocidade da navegação, diz se a região ainda estará no quadro."""
    ajustar_luz: bool = True
    """Com luz auxiliar no hardware: mais luz no quadro escuro, menos no estourado (e volta
    ao nível de antes depois das vistas)."""
    passo_luz: float = 0.3
    incerto_em_novidade: bool = True
    """Recorte longe de tudo o que a cabeça conhece (referência ``.novidade.npz``) conta como
    dúvida **de decisão**, mesmo com confiança alta: a confiança da cabeça não vale ali."""
    limiar_novidade: float = 1.2
    """Escore de novidade (em limiares da referência) acima do qual o recorte está fora do que a
    cabeça conhece. Não é 1: folhas de café que a cabeça não viu (BRACOL separado), passadas
    pelo caminho do quadro (redimensiona + JPEG), chegam a 1,105 — com 1 elas sairiam 31% "fora".
    Medido em 19/10/2026: com 1,2, 0 de 48 dessas folhas; no vídeo do YouTube, 46% dos quadros
    (quase tudo cena que não é folha). Recalibrar com quadro de Rover em cafezal."""
    fracao_recortes_novos: float = 0.3
    """Fração dos recortes fora da distribuição que basta para o quadro; um recorte que decide
    a doença e está fora já basta."""
    usar_termica: bool = True
    """Com câmera térmica no hardware: no caso ambíguo ou sem informação suficiente, mede a
    temperatura da região contra o quadro (evidência ao agrônomo, não decide)."""

    def validate(self) -> None:
        if self.max_vistas_extras < 0 or self.max_extras_por_minuto < 0:
            raise ConfigError("percepcao_ativa: limites de vistas não podem ser negativos.")
        if self.tempo_max_ms <= 0:
            raise ConfigError("percepcao_ativa.tempo_max_ms precisa ser positivo.")
        if self.margem_evidencia is not None and self.margem_evidencia < 0:
            raise ConfigError("percepcao_ativa.margem_evidencia negativa.")
        if not 0.0 <= self.min_registro <= 1.0:
            raise ConfigError("percepcao_ativa.min_registro fora de [0, 1].")
        if not (0.0 <= self.max_desvio_vertical <= 1.0
                and 0.0 <= self.max_desvio_horizontal <= 1.0):
            raise ConfigError("percepcao_ativa: max_desvio_* em fração da foto, de 0 a 1.")
        if self.margem_regiao < 0 or self.pedir_pausa_s < 0:
            raise ConfigError("percepcao_ativa: margem e pausa não podem ser negativas.")
        if self.guardar_vistas not in ("incertas", "sempre", "nunca"):
            raise ConfigError("percepcao_ativa.guardar_vistas: use incertas, sempre ou nunca.")
        if not (0.0 <= self.prioridade_minima <= self.prioridade_alta <= 1.0):
            raise ConfigError("percepcao_ativa: 0 <= prioridade_minima <= prioridade_alta <= 1.")
        if self.reserva_prioridade_alta < 0:
            raise ConfigError("percepcao_ativa.reserva_prioridade_alta negativa.")
        if self.min_recortes_ambiguos < 1:
            raise ConfigError("percepcao_ativa.min_recortes_ambiguos precisa ser ao menos 1.")
        if not 0.0 < self.fracao_qualidade_baixa <= 1.0:
            raise ConfigError("percepcao_ativa.fracao_qualidade_baixa fora de (0, 1].")
        if not 0.0 < self.fator_velocidade < 1.0:
            raise ConfigError("percepcao_ativa.fator_velocidade em (0, 1): reduzir, não parar.")
        if self.duracao_reducao_s <= 0 or self.latencia_vista_s <= 0:
            raise ConfigError("percepcao_ativa: duração da redução e latência positivas.")
        if not 0.0 < self.passo_luz <= 1.0:
            raise ConfigError("percepcao_ativa.passo_luz fora de (0, 1].")
        if self.limiar_novidade <= 0:
            raise ConfigError("percepcao_ativa.limiar_novidade precisa ser positivo.")
        if not 0.0 < self.fracao_recortes_novos <= 1.0:
            raise ConfigError("percepcao_ativa.fracao_recortes_novos fora de (0, 1].")


PRIORIDADES_CORTAVEIS = ("DOENCA_PRAGA", "ANOMALIA", "CONCLUSAO_MISSAO", "NORMAL_COMPACTADA",
                         "SECUNDARIO")


def _fracoes_padrao() -> dict[str, float]:
    # Arquivo nenhum passa de 60% do teto: os 40% de cima ficam para os resultados (os
    # metadados e recortes que a fusão por planta usa para contar TODAS as plantas) e para o
    # que nunca corta. Simulado: sem essa reserva, os originais da manhã comem o dia inteiro.
    return {"DOENCA_PRAGA": 0.6, "ANOMALIA": 0.5, "CONCLUSAO_MISSAO": 0.6,
            "NORMAL_COMPACTADA": 0.4, "SECUNDARIO": 0.3}


@dataclass(frozen=True, slots=True)
class DataBudgetConfig:
    """Orçamento de dados pela Starlink do Rover (:mod:`valtexa_rover.runtime.orcamento`).

    O teto conta **subida e descida** (o plano da Starlink cobra as duas), mas só a subida é
    cortável. Nunca cortam: comando, missão, telemetria, eventos (registro ``event``) e
    qualquer item de prioridade 1 (evento crítico). O resto sobe enquanto o usado do dia (e
    da missão) estiver abaixo da fração do teto da sua prioridade; acima, **espera** o dia
    seguinte na fila, ou vira amostra (``amostra_adiados_cada``).

    Os padrões são **provisórios**: o teto certo é o do plano Starlink contratado (decisão do
    Pedro). A plataforma pode mandar outro teto na missão assinada — esse vale por cima.
    """

    bytes_dia: int = 2_000_000_000
    """Teto do dia (0 = sem teto). 2 GB/dia ≈ 60 GB/mês — provisório."""
    bytes_missao: int = 0
    """Teto da missão (0 = sem teto); vale junto com o do dia (o que acabar primeiro)."""
    fuso_horas: float = -3.0
    """O "dia" do orçamento é o da fazenda (Brasília), não o UTC."""
    fracao_por_prioridade: dict[str, float] = field(default_factory=_fracoes_padrao)
    """Até que fração do teto os **arquivos** de cada prioridade podem subir. Os resultados
    (metadados) da Edge AI usam 1,0: só esperam com o teto inteiro gasto."""
    sobrecarga_requisicao_bytes: int = 700
    """Cabeçalhos HTTP + TLS por requisição, estimados (a Starlink cobra o que trafega, não
    só o corpo)."""
    amostra_adiados_cada: int = 10
    """Item cortável (prioridade ≥ 5) que esperou mais que ``adiado_max_h``: fica 1 a cada N
    na fila; o resto é retido no disco (não sobe). 0 = só espera."""
    adiado_max_h: float = 24.0
    tempestade_janela_s: float = 120.0
    tempestade_min_quadros: int = 40
    """Com menos quadros na janela não se julga tempestade (começo de missão)."""
    tempestade_fracao: float = 0.5
    """Fração de quadros em evento na janela que liga a tempestade."""
    tempestade_fracao_saida: float = 0.25
    """Abaixo disto ela desliga (histerese: não pisca na borda)."""
    tempestade_originais_por_janela: int = 4
    """Originais de evento que ainda sobem por janela durante a tempestade (o resto fica no
    disco, retido). Os quadros de contexto do buffer não sobem durante a tempestade."""
    k_por_planta: int = 2
    """Originais de evento por planta (os k de maior confiança). 0 desliga a deduplicação."""
    espacamento_planta_m: float = 0.7
    """Planta = (rua, metro ao longo da rua / espaçamento). Sem a navegação dizer a rua e o
    metro, não há como saber a planta, e não se deduplica."""

    def fracao(self, prioridade: int) -> float:
        if prioridade <= int(Priority.EVENTO_CRITICO):
            return float("inf")
        return self.fracao_por_prioridade.get(Priority(prioridade).name, 1.0)

    def validate(self) -> None:
        if self.bytes_dia < 0 or self.bytes_missao < 0:
            raise ConfigError("orcamento_dados: tetos não podem ser negativos.")
        if not -14 <= self.fuso_horas <= 14:
            raise ConfigError("orcamento_dados.fuso_horas fora de [-14, 14].")
        sobra = set(self.fracao_por_prioridade) - set(PRIORIDADES_CORTAVEIS)
        if sobra:
            raise ConfigError(f"orcamento_dados.fracao_por_prioridade: prioridade(s) "
                              f"desconhecida(s) {sorted(sobra)} (use {PRIORIDADES_CORTAVEIS}).")
        anterior = 1.0
        for nome in PRIORIDADES_CORTAVEIS:
            f = self.fracao_por_prioridade.get(nome, 1.0)
            if not 0.0 <= f <= 1.0:
                raise ConfigError(f"orcamento_dados.fracao_por_prioridade.{nome} fora de [0, 1].")
            if nome != "CONCLUSAO_MISSAO" and f > anterior + 1e-12:
                raise ConfigError("orcamento_dados.fracao_por_prioridade: a prioridade menos "
                                  f"urgente ({nome}) não pode ter fração maior que a anterior.")
            if nome != "CONCLUSAO_MISSAO":
                anterior = f
        if self.sobrecarga_requisicao_bytes < 0 or self.amostra_adiados_cada < 0:
            raise ConfigError("orcamento_dados: sobrecarga e amostra não podem ser negativas.")
        if self.adiado_max_h <= 0 or self.tempestade_janela_s <= 0:
            raise ConfigError("orcamento_dados: adiado_max_h e tempestade_janela_s positivos.")
        if self.tempestade_min_quadros < 1:
            raise ConfigError("orcamento_dados.tempestade_min_quadros precisa ser ao menos 1.")
        if not 0 < self.tempestade_fracao_saida <= self.tempestade_fracao <= 1:
            raise ConfigError("orcamento_dados: 0 < tempestade_fracao_saida <= "
                              "tempestade_fracao <= 1.")
        if self.tempestade_originais_por_janela < 0 or self.k_por_planta < 0:
            raise ConfigError("orcamento_dados: originais por janela e k não negativos.")
        if self.espacamento_planta_m < 0:
            raise ConfigError("orcamento_dados.espacamento_planta_m negativo.")


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    classificacao: ClassificationConfig = field(default_factory=ClassificationConfig)
    retencao: RetentionConfig = field(default_factory=RetentionConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    saude: HealthConfig = field(default_factory=HealthConfig)
    profundidade: DepthConfig = field(default_factory=DepthConfig)
    percepcao_ativa: ActivePerceptionConfig = field(default_factory=ActivePerceptionConfig)
    orcamento_dados: DataBudgetConfig = field(default_factory=DataBudgetConfig)

    def validate(self) -> RuntimeConfig:
        self.classificacao.validate()
        self.retencao.validate()
        self.buffer.validate()
        self.sync.validate()
        self.saude.validate()
        self.profundidade.validate()
        self.percepcao_ativa.validate()
        self.orcamento_dados.validate()
        return self

    # -- JSON ---------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RuntimeConfig:
        _sem_desconhecidas("raiz", data, {f.name for f in fields(cls)})
        return cls(
            classificacao=_dc(ClassificationConfig, data.get("classificacao"), "classificacao"),
            retencao=_retencao(data.get("retencao")),
            buffer=_dc(BufferConfig, data.get("buffer"), "buffer", {"priority": Priority}),
            sync=_dc(SyncConfig, data.get("sync"), "sync"),
            saude=_dc(HealthConfig, data.get("saude"), "saude"),
            profundidade=_dc(DepthConfig, data.get("profundidade"), "profundidade"),
            percepcao_ativa=_dc(ActivePerceptionConfig, data.get("percepcao_ativa"),
                                "percepcao_ativa"),
            orcamento_dados=_orcamento(data.get("orcamento_dados")),
        ).validate()

    @classmethod
    def load(cls, path: Path | str | None) -> RuntimeConfig:
        if path is None or not Path(path).exists():
            return cls().validate()
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{path}: JSON inválido ({exc}).") from exc
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        bruto = asdict(self)
        bruto["retencao"]["rules"] = {
            str(k): {**asdict(v), "tier": v.tier.value, "priority": int(v.priority)}
            for k, v in self.retencao.rules.items()
        }
        bruto["buffer"]["priority"] = int(self.buffer.priority)
        return bruto


def _sem_desconhecidas(secao: str, data: dict, conhecidas: set[str]) -> None:
    sobra = set(data) - conhecidas
    if sobra:
        raise ConfigError(f"{secao}: chave(s) desconhecida(s) {sorted(sobra)}.")


def _dc(tipo, data: dict | None, secao: str, conversores: dict | None = None):
    if data is None:
        return tipo()
    if not isinstance(data, dict):
        raise ConfigError(f"{secao}: esperava um objeto.")
    _sem_desconhecidas(secao, data, {f.name for f in fields(tipo)})
    valores = dict(data)
    for chave, conv in (conversores or {}).items():
        if chave in valores:
            valores[chave] = conv(valores[chave])
    try:
        return tipo(**valores)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{secao}: {exc}") from exc


def _retencao(data: dict | None) -> RetentionConfig:
    if data is None:
        return RetentionConfig()
    _sem_desconhecidas("retencao", data, {"rules", "normal_sample_every"})
    regras = _regras_padrao()
    nomes_validos = {c.value for c in Category} | {"BAIXA_CONFIANCA"}
    for nome, bruto in (data.get("rules") or {}).items():
        if nome not in nomes_validos:
            raise ConfigError(f"retencao.rules: categoria desconhecida {nome!r}.")
        base = regras[Category(nome) if nome != "BAIXA_CONFIANCA" else nome]
        _sem_desconhecidas(f"retencao.rules.{nome}", bruto, {f.name for f in fields(RetentionRule)})
        try:
            valores = {**asdict(base), **bruto}
            valores["tier"] = Tier(valores["tier"])
            valores["priority"] = Priority(valores["priority"])
        except ValueError as exc:
            raise ConfigError(f"retencao.rules.{nome}: {exc}") from exc
        chave = Category(nome) if nome != "BAIXA_CONFIANCA" else nome
        regras[chave] = RetentionRule(**valores)
    amostra = int(data.get("normal_sample_every", 0))
    return RetentionConfig(rules=regras, normal_sample_every=amostra)


def _orcamento(data: dict | None) -> DataBudgetConfig:
    """As frações por prioridade se mesclam com as padrão (mandar uma não apaga as outras)."""
    if isinstance(data, dict) and isinstance(data.get("fracao_por_prioridade"), dict):
        data = {**data, "fracao_por_prioridade": {**_fracoes_padrao(),
                                                  **data["fracao_por_prioridade"]}}
    return _dc(DataBudgetConfig, data, "orcamento_dados")
