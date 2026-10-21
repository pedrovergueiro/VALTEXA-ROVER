"""Orçamento de dados: o que sobe pela Starlink do Rover, quanto, e o que espera.

Desde a rodada 4 não há base na fazenda: cada Rover fala com a plataforma pela **sua**
Starlink, e o volume enviado é requisito (``VALTEXA-SAAS/docs/historico/RODADA4.md``).
Quatro peças, todas daqui:

* **Contabilidade** (:meth:`OrcamentoDados.contar`): toda requisição do
  :class:`~valtexa_rover.runtime.client.DeviceClient` — corpo enviado, corpo recebido e a
  sobrecarga de cabeçalho/TLS estimada — vai para a tabela ``uso_dados`` por dia da fazenda,
  missão e classe (``registros``, ``arquivos``, ``telemetria``, ``comando``, ``missao``,
  ``atualizacao``, ``video``, ``outros``). É o que de fato trafegou, não o que se pretendia.
* **Corte por prioridade** (:meth:`OrcamentoDados.prioridade_maxima`, usada pela fila do
  :class:`~valtexa_rover.runtime.sync.SyncEngine`): cada prioridade sobe enquanto o usado do
  dia (e da missão) estiver abaixo da fração do teto dela. **Nunca cortam**: comando, missão,
  telemetria, eventos e prioridade 1 — esses saem mesmo com o teto estourado (e contam).
  O que é cortado **espera** na fila (dia novo, teto novo); o cortável que esperou demais
  vira amostra (:meth:`OrcamentoDados.amostrar_adiados`): 1 a cada N continua na fila, o
  resto fica no disco, retido.
* **Tempestade de eventos** (:class:`DetectorDeTempestade`): fração de quadros em evento
  numa janela acima do normal — o modelo que dispara em tudo (visto em vídeo real: 511/511
  quadros). Enquanto durar: poucos originais por janela sobem, os quadros de contexto do
  buffer não sobem, o resto fica no disco **retido**, e a plataforma recebe ``EVENT_STORM``
  (prioridade 1, nunca cortado).
* **Deduplicação por planta** (:class:`DeduplicadorPorPlanta`): a mesma planta é vista por
  várias câmeras e em quadros seguidos; só os ``k`` originais de maior confiança sobem. O
  que sai do pódio é retido (se ainda não começou a subir).

**Retido não é apagado.** O item fica na fila com status ``RETIDO`` e o motivo, o arquivo no
disco (com a retenção de sempre). A plataforma pode pedir para liberar
(:meth:`OrcamentoDados.liberar_retidos`). O que nunca é retido: o registro do resultado (os
metadados e os recortes que a fusão por planta usa) — a incidência continua contando todas as
plantas, mesmo na tempestade.

O teto vem do ``rover.json`` (``orcamento_dados``) e, quando a missão assinada traz
``orcamento_dados``, o da plataforma vale por cima (:meth:`OrcamentoDados.aplicar_da_plataforma`).
"""

from __future__ import annotations

import contextlib
from collections import OrderedDict, deque
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any

from valtexa_rover.clock import Clock
from valtexa_rover.config import DataBudgetConfig, Priority
from valtexa_rover.runtime.store import LocalStore

CLASSES = ("registros", "arquivos", "telemetria", "comando", "missao", "atualizacao", "video",
           "outros")
NUNCA_CORTAM = ("comando", "missao", "telemetria", "atualizacao", "video")
"""Classes que a fila do orçamento não segura (a OTA e o vídeo ao vivo têm controle próprio:
a OTA só fora de missão, o vídeo só sob pedido da equipe). Contam no usado."""
RETIDO = "RETIDO"
MOTIVOS = ("tempestade", "tempestade_contexto", "dedup_planta", "amostra_orcamento")
CHAVE_PLATAFORMA = "orcamento_plataforma"
NIVEIS_ALERTA = (0.8, 1.0)


def classe_do_caminho(path: str) -> str:
    """A classe de uma requisição pelo caminho do protocolo do dispositivo."""
    p = path.partition("?")[0]
    if p.startswith("/v1/device/edge/records"):
        return "registros"
    if p.startswith("/v1/device/uploads"):
        return "arquivos"
    if p.startswith("/v1/device/sync"):
        return "telemetria"
    if p.startswith(("/v1/device/commands", "/v1/device/teleop")):
        return "comando"
    if p.startswith(("/v1/device/missions", "/v1/device/session")):
        return "missao"
    if p.startswith(("/v1/device/ota", "/v1/device/models", "/v1/device/artifacts",
                     "/v1/device/priores")):
        return "atualizacao"
    if p.startswith("/v1/device/video"):
        return "video"
    return "outros"


# ---------------------------------------------------------------------------------------------
# Tempestade de eventos
# ---------------------------------------------------------------------------------------------


@dataclass
class DetectorDeTempestade:
    """Janela deslizante (tempo monotônico) da fração de quadros em evento, todas as câmeras.

    Liga com ``fracao`` (e pelo menos ``min_quadros`` na janela), desliga abaixo de
    ``fracao_saida``. Na tempestade, :meth:`permite_original` deixa passar
    ``originais_por_janela`` originais de evento por janela; os outros ficam retidos.
    """

    janela_s: float
    min_quadros: int
    fracao: float
    fracao_saida: float
    originais_por_janela: int
    ativa: bool = False
    inicio: float | None = None
    _quadros: deque = field(default_factory=deque)
    _originais: deque = field(default_factory=deque)
    retidos: int = 0

    @classmethod
    def de(cls, cfg: DataBudgetConfig) -> DetectorDeTempestade:
        return cls(cfg.tempestade_janela_s, cfg.tempestade_min_quadros, cfg.tempestade_fracao,
                   cfg.tempestade_fracao_saida, cfg.tempestade_originais_por_janela)

    def _limpa(self, agora: float) -> None:
        while self._quadros and agora - self._quadros[0][0] > self.janela_s:
            self._quadros.popleft()
        while self._originais and agora - self._originais[0] > self.janela_s:
            self._originais.popleft()

    def taxa(self) -> float:
        if not self._quadros:
            return 0.0
        return sum(1 for _, e in self._quadros if e) / len(self._quadros)

    def observa(self, evento: bool, agora: float) -> str | None:
        """Registra um quadro. Devolve ``"inicio"`` ou ``"fim"`` quando o estado muda."""
        self._quadros.append((agora, bool(evento)))
        self._limpa(agora)
        if len(self._quadros) < self.min_quadros:
            return None
        taxa = self.taxa()
        if not self.ativa and taxa >= self.fracao:
            self.ativa, self.inicio, self.retidos = True, agora, 0
            self._originais.clear()
            return "inicio"
        if self.ativa and taxa < self.fracao_saida:
            self.ativa = False
            return "fim"
        return None

    def permite_original(self, agora: float) -> bool:
        if not self.ativa:
            return True
        self._limpa(agora)
        if len(self._originais) < self.originais_por_janela:
            self._originais.append(agora)
            return True
        self.retidos += 1
        return False

    def estado(self) -> dict[str, Any]:
        return {"ativa": self.ativa, "taxa_eventos": round(self.taxa(), 4),
                "quadros_na_janela": len(self._quadros), "janela_s": self.janela_s,
                "retidos_nesta": self.retidos}


# ---------------------------------------------------------------------------------------------
# Deduplicação por planta
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EntradaPlanta:
    escore: float
    record_ids: tuple[str, ...]
    """O resultado e o evento: os arquivos ligados a eles (original, vistas, contexto)."""


class DeduplicadorPorPlanta:
    """Os ``k`` melhores originais de evento por planta (rua, índice da planta na rua).

    A planta sai da navegação: ``row_left``/``row_right`` pelo lado da câmera e ``x_m`` ao
    longo da rua, dividido pelo espaçamento. Sem isso, :meth:`chave` devolve ``None`` e o
    quadro não é deduplicado (não há como saber que é a mesma planta). A memória é limitada
    (``max_plantas``, a mais antiga sai) e zera a cada missão.
    """

    def __init__(self, k: int, espacamento_m: float, max_plantas: int = 50_000) -> None:
        self.k = k
        self.espacamento_m = espacamento_m
        self.max_plantas = max_plantas
        self._plantas: OrderedDict[tuple, list[EntradaPlanta]] = OrderedDict()

    @property
    def ligado(self) -> bool:
        return self.k > 0 and self.espacamento_m > 0

    def chave(self, nav: dict | None, lado: str | None) -> tuple | None:
        if not self.ligado or not nav or nav.get("x_m") is None:
            return None
        rua = nav.get("row_left") if lado == "left" else (
            nav.get("row_right") if lado == "right" else None)
        if rua is None:
            return None
        return (str(rua), round(float(nav["x_m"]) / self.espacamento_m))

    def decide(self, chave: tuple, escore: float, record_ids: tuple[str, ...]
               ) -> tuple[bool, list[EntradaPlanta]]:
        """``(sobe, rebaixados)``: se este original sobe e quem saiu do pódio por causa dele."""
        lista = self._plantas.get(chave)
        if lista is None:
            lista = self._plantas[chave] = []
            while len(self._plantas) > self.max_plantas:
                self._plantas.popitem(last=False)
        else:
            self._plantas.move_to_end(chave)
        nova = EntradaPlanta(float(escore), record_ids)
        if len(lista) < self.k:
            lista.append(nova)
            return True, []
        pior = min(lista, key=lambda e: e.escore)
        if nova.escore <= pior.escore:
            return False, []
        lista.remove(pior)
        lista.append(nova)
        return True, [pior]

    def reinicia(self) -> None:
        self._plantas.clear()


# ---------------------------------------------------------------------------------------------
# O orçamento
# ---------------------------------------------------------------------------------------------


class OrcamentoDados:
    def __init__(self, store: LocalStore, clock: Clock, config: DataBudgetConfig,
                 events=None) -> None:
        self.store = store
        self.clock = clock
        self.base = config
        self.events = events
        self.tempestade = DetectorDeTempestade.de(config)
        self.dedup = DeduplicadorPorPlanta(config.k_por_planta, config.espacamento_planta_m)
        self.missao_ativa: str | None = None
        self._missao_envio: str | None = None
        self._eventos_retidos: dict[str, str] = {}
        with store.transaction() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS uso_dados (dia TEXT NOT NULL, missao TEXT NOT NULL, "
                "classe TEXT NOT NULL, enviados INTEGER NOT NULL DEFAULT 0, "
                "recebidos INTEGER NOT NULL DEFAULT 0, requisicoes INTEGER NOT NULL DEFAULT 0, "
                "PRIMARY KEY (dia, missao, classe))")

    # -- configuração efetiva -------------------------------------------------------------

    @property
    def config(self) -> DataBudgetConfig:
        """O ``rover.json`` com o teto da plataforma (missão assinada) por cima, se houver."""
        plat = self.store.get(CHAVE_PLATAFORMA) or {}
        mudancas = {k: int(plat[k]) for k in ("bytes_dia", "bytes_missao")
                    if isinstance(plat.get(k), int) and plat[k] >= 0}
        return replace(self.base, **mudancas) if mudancas else self.base

    def aplicar_da_plataforma(self, corpo: dict | None) -> bool:
        """Guarda o teto que veio na missão assinada (``orcamento_dados``). Inválido: ignora."""
        if not isinstance(corpo, dict):
            return False
        valido = {k: corpo[k] for k in ("bytes_dia", "bytes_missao")
                  if isinstance(corpo.get(k), int) and not isinstance(corpo.get(k), bool)
                  and corpo[k] >= 0}
        if not valido:
            return False
        self.store.put(CHAVE_PLATAFORMA, {**valido, "recebido_em": self.clock.now().isoformat()})
        return True

    def dia(self, quando: datetime | None = None) -> str:
        quando = quando or self.clock.now()
        if quando.tzinfo is None:
            quando = quando.replace(tzinfo=timezone.utc)
        fuso = timezone(timedelta(hours=self.base.fuso_horas))
        return quando.astimezone(fuso).date().isoformat()

    # -- contabilidade ----------------------------------------------------------------------

    @contextlib.contextmanager
    def atribuir(self, mission_id: str | None) -> Iterator[None]:
        """As requisições dentro do bloco contam para esta missão (a do item que sobe)."""
        anterior, self._missao_envio = self._missao_envio, mission_id
        try:
            yield
        finally:
            self._missao_envio = anterior

    def contar(self, method: str, path: str, enviados: int, recebidos: int) -> None:
        """O gancho do ``DeviceClient``: uma requisição que de fato trafegou."""
        classe = classe_do_caminho(path)
        missao = self._missao_envio or self.missao_ativa or ""
        extra = self.base.sobrecarga_requisicao_bytes
        dia = self.dia()
        antes = self.usado_dia(dia)
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO uso_dados (dia, missao, classe, enviados, recebidos, requisicoes) "
                "VALUES (?, ?, ?, ?, ?, 1) ON CONFLICT (dia, missao, classe) DO UPDATE SET "
                "enviados = enviados + excluded.enviados, "
                "recebidos = recebidos + excluded.recebidos, requisicoes = requisicoes + 1",
                (dia, missao, classe, int(enviados) + extra, int(recebidos)))
        self._alerta_se_cruzou(dia, antes, self.usado_dia(dia))

    def usado_dia(self, dia: str | None = None) -> int:
        return int(self.store.scalar(
            "SELECT COALESCE(SUM(enviados + recebidos), 0) FROM uso_dados WHERE dia = ?",
            (dia or self.dia(),)) or 0)

    def usado_missao(self, mission_id: str) -> int:
        return int(self.store.scalar(
            "SELECT COALESCE(SUM(enviados + recebidos), 0) FROM uso_dados WHERE missao = ?",
            (mission_id,)) or 0)

    # -- corte ------------------------------------------------------------------------------

    def _fracao_usada(self, mission_id: str | None) -> float:
        cfg = self.config
        usos = []
        if cfg.bytes_dia:
            usos.append(self.usado_dia() / cfg.bytes_dia)
        if cfg.bytes_missao and mission_id:
            usos.append(self.usado_missao(mission_id) / cfg.bytes_missao)
        return max(usos, default=0.0)

    def prioridade_maxima(self, mission_id: str | None = None) -> int:
        """A prioridade menos urgente que ainda sobe agora (1 = só o que nunca corta)."""
        cfg = self.config
        usada = self._fracao_usada(mission_id)
        liberada = int(Priority.EVENTO_CRITICO)
        for p in range(int(Priority.DOENCA_PRAGA), int(Priority.SECUNDARIO) + 1):
            if usada < cfg.fracao(p):
                liberada = p
        return liberada

    def metadados_liberados(self, mission_id: str | None = None) -> bool:
        """Resultados (metadados) só esperam com o teto inteiro gasto."""
        return self._fracao_usada(mission_id) < 1.0

    def adiados(self) -> int:
        """Itens na fila que o orçamento está segurando agora."""
        limite = self.prioridade_maxima()
        meta = self.metadados_liberados()
        return int(self.store.scalar(
            "SELECT COUNT(*) FROM sync_queue q LEFT JOIN records r ON r.id = q.ref_id "
            "WHERE q.status IN ('PENDENTE', 'ENVIANDO') AND q.priority > 1 AND ("
            "(q.kind = 'file' AND q.priority > ?) OR "
            "(q.kind = 'record' AND COALESCE(r.kind, '') != 'event' AND ? = 0))",
            (limite, 1 if meta else 0)) or 0)

    # -- retenção (tempestade, deduplicação, amostra) --------------------------------------

    def observa_quadro(self, evento: bool) -> str | None:
        """Cada quadro passa aqui; avisa a plataforma quando a tempestade começa ou acaba."""
        mudou = self.tempestade.observa(evento, self.clock.monotonic())
        if mudou and self.events is not None:
            from valtexa_rover.runtime.events import EventType

            self.events.emit(EventType.EVENT_STORM, {"estado": mudou, **self.tempestade.estado()},
                             mission_id=self.missao_ativa, severity="warning",
                             priority=Priority.EVENTO_CRITICO)
        return mudou

    def motivo_de_reter(self, chave_planta: tuple | None, escore: float,
                        record_ids: tuple[str, ...]) -> tuple[str | None, list[EntradaPlanta]]:
        """Para o original de um evento: ``(motivo ou None, rebaixados pela deduplicação)``."""
        rebaixados: list[EntradaPlanta] = []
        if chave_planta is not None:
            sobe, rebaixados = self.dedup.decide(chave_planta, escore, record_ids)
            if not sobe:
                return "dedup_planta", []
        if not self.tempestade.permite_original(self.clock.monotonic()):
            return "tempestade", rebaixados
        return None, rebaixados

    def marca_evento(self, event_id: str, motivo: str | None) -> None:
        """O contexto do buffer deste evento segue o original (retido se ele foi)."""
        if motivo:
            self._eventos_retidos[event_id] = motivo
        elif self.tempestade.ativa:
            self._eventos_retidos[event_id] = "tempestade_contexto"

    def motivo_do_contexto(self, event_id: str) -> str | None:
        return self._eventos_retidos.pop(event_id, None)

    def reter(self, conn, record_ids: tuple[str, ...], motivo: str) -> int:
        """Retém (se ainda não começaram a subir) os arquivos ligados a estes registros."""
        if not record_ids:
            return 0
        marcas = ",".join("?" * len(record_ids))
        cur = conn.execute(
            f"UPDATE sync_queue SET status = '{RETIDO}', last_error = ? WHERE kind = 'file' "
            "AND status = 'PENDENTE' AND chunks_done = 0 AND priority > 1 AND ref_id IN "
            f"(SELECT id FROM files WHERE record_id IN ({marcas}))",
            (f"retido: {motivo}", *record_ids))
        for rid in record_ids:
            if rid in self._eventos_retidos or motivo == "dedup_planta":
                self._eventos_retidos[rid] = motivo
        return cur.rowcount

    def amostrar_adiados(self) -> int:
        """Cortável (prioridade ≥ 5) esperando há mais de ``adiado_max_h``: 1 a cada N fica."""
        n = self.base.amostra_adiados_cada
        if not n:
            return 0
        limite = (self.clock.now() - timedelta(hours=self.base.adiado_max_h)).isoformat()
        linhas = self.store.query(
            "SELECT id FROM sync_queue WHERE kind = 'file' AND status = 'PENDENTE' "
            "AND chunks_done = 0 AND priority >= ? AND created_at < ? ORDER BY seq",
            (int(Priority.NORMAL_COMPACTADA), limite))
        retidos = [linha["id"] for i, linha in enumerate(linhas) if i % n != 0]
        with self.store.transaction() as conn:
            conn.executemany(
                f"UPDATE sync_queue SET status = '{RETIDO}', last_error = ? WHERE id = ?",
                [("retido: amostra_orcamento", q) for q in retidos])
        return len(retidos)

    def liberar_retidos(self, *, motivo: str | None = None, record_ids: list[str] | None = None,
                        maximo: int = 100) -> int:
        """A plataforma pediu: volta à fila (``PENDENTE``) o que estava retido."""
        sql = f"SELECT q.id FROM sync_queue q JOIN files f ON f.id = q.ref_id " \
              f"WHERE q.status = '{RETIDO}'"
        params: list[Any] = []
        if motivo:
            sql += " AND q.last_error = ?"
            params.append(f"retido: {motivo}")
        if record_ids:
            sql += f" AND f.record_id IN ({','.join('?' * len(record_ids))})"
            params += list(record_ids)
        sql += " ORDER BY q.priority, q.seq LIMIT ?"
        params.append(int(maximo))
        ids = [linha["id"] for linha in self.store.query(sql, params)]
        with self.store.transaction() as conn:
            conn.executemany("UPDATE sync_queue SET status = 'PENDENTE', last_error = NULL "
                             "WHERE id = ?", [(q,) for q in ids])
        return len(ids)

    def nova_missao(self, mission_id: str | None) -> None:
        self.missao_ativa = mission_id
        self.dedup.reinicia()
        self._eventos_retidos.clear()

    # -- relatório --------------------------------------------------------------------------

    def relatorio(self, dia: str | None = None) -> dict[str, Any]:
        """O consumo do dia por classe, o teto, o que espera e o que ficou retido (e por quê)."""
        dia = dia or self.dia()
        cfg = self.config
        por_classe = {
            linha["classe"]: {"enviados": int(linha["e"]), "recebidos": int(linha["r"]),
                              "requisicoes": int(linha["n"])}
            for linha in self.store.query(
                "SELECT classe, SUM(enviados) AS e, SUM(recebidos) AS r, SUM(requisicoes) AS n "
                "FROM uso_dados WHERE dia = ? GROUP BY classe", (dia,))}
        retidos = {
            (linha["m"] or "").removeprefix("retido: "): {"itens": int(linha["n"]),
                                                          "bytes": int(linha["b"] or 0)}
            for linha in self.store.query(
                f"SELECT last_error AS m, COUNT(*) AS n, SUM(size_bytes) AS b FROM sync_queue "
                f"WHERE status = '{RETIDO}' GROUP BY last_error")}
        usado = self.usado_dia(dia)
        plat = self.store.get(CHAVE_PLATAFORMA)
        return {
            "dia": dia, "usado_bytes": usado,
            "enviados_bytes": sum(c["enviados"] for c in por_classe.values()),
            "recebidos_bytes": sum(c["recebidos"] for c in por_classe.values()),
            "teto_dia_bytes": cfg.bytes_dia or None, "teto_missao_bytes": cfg.bytes_missao or None,
            "fracao_usada": round(usado / cfg.bytes_dia, 4) if cfg.bytes_dia else None,
            "origem_do_teto": "plataforma" if plat else "rover.json",
            "prioridade_maxima_liberada": self.prioridade_maxima(self.missao_ativa),
            "por_classe": por_classe, "adiados": self.adiados(), "retidos": retidos,
            "tempestade": self.tempestade.estado(),
        }

    def _alerta_se_cruzou(self, dia: str, antes: int, depois: int) -> None:
        cfg = self.config
        if self.events is None or not cfg.bytes_dia:
            return
        from valtexa_rover.runtime.events import EventType

        for nivel in NIVEIS_ALERTA:
            marca = nivel * cfg.bytes_dia
            if antes < marca <= depois:
                chave = f"orcamento_alerta:{dia}:{nivel}"
                if self.store.get(chave):
                    continue
                self.store.put(chave, True)
                self.events.emit(EventType.DATA_BUDGET,
                                 {"nivel": nivel, "dia": dia, "usado_bytes": depois,
                                  "teto_dia_bytes": cfg.bytes_dia,
                                  "prioridade_maxima_liberada": self.prioridade_maxima()},
                                 mission_id=self.missao_ativa,
                                 severity="warning" if nivel < 1 else "critical",
                                 priority=Priority.EVENTO_CRITICO)
