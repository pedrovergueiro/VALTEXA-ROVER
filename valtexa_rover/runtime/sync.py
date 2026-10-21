"""Sync Engine: fila persistente, chunks, retomada, prioridade e *backoff* (seções 18, 19, 52).

Cada item da fila (tabela ``sync_queue``) tem ID, hash, tamanho, chunks,
progresso, prioridade, status, tentativas e horário — os campos da seção 18.
Dois tipos:

* ``record`` — um resultado da Edge AI ou um evento. Sobe em lote por
  ``POST /v1/device/edge/records``; a nuvem deduplica por ``(id, sha256)``;
* ``file`` — uma imagem (original, reduzida ou de contexto). Sobe em chunks:
  ``POST /v1/device/uploads`` abre (ou reabre) o upload e diz **quais chunks a
  nuvem já tem**; o robô manda só os que faltam (``PUT .../chunks/{i}``) e fecha
  (``POST .../complete``), quando a nuvem confere o SHA-256 do arquivo inteiro.
  Caiu em 70 %? O progresso de cada chunk foi gravado; na volta, a nuvem
  responde os 70 % que já tem e o envio continua dali. Reenviar o mesmo
  arquivo (mesmo hash) não cria outro objeto.

Telemetria vai à parte, pelo ``POST /v1/device/sync`` que já existia, com o
``remaining`` (quantos itens ainda esperam) — é o que faz a frota mostrar
SINCRONIZANDO.

Ordem: prioridade da seção 19 (1 = evento crítico … 6 = secundário), depois a
ordem de criação. Erros:

* sem rede → o item não perde tentativa; o passo termina e tenta depois;
* ``5xx``, ``408``, ``429``, ``401`` (relógio) → *backoff* exponencial com
  *jitter* (``backoff_base_s`` · 2^tentativas, até ``backoff_max_s``);
* ``403`` → para a sincronização inteira (protocolo, seção 7: nunca insistir);
* outro ``4xx`` (inclusive ``409``: mesmo ID com conteúdo diferente) → FALHOU,
  com ``UPLOAD_FAILED``. Não adianta repetir o que a nuvem recusou pelo conteúdo.

**Orçamento de dados** (:mod:`valtexa_rover.runtime.orcamento`): com ele, a seleção só pega
o que o teto deixa subir agora — prioridade 1 e eventos sempre; resultados enquanto o teto
não acabou; arquivos até a fração da prioridade deles. O resto **espera** na fila (não perde
tentativa, não falha). Itens ``RETIDO`` (tempestade, deduplicação, amostra) não sobem até a
plataforma pedir.
"""

from __future__ import annotations

import contextlib
import json
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from valtexa_rover import EDGE_AI_VERSION, RUNTIME_VERSION
from valtexa_rover.clock import Clock
from valtexa_rover.config import Priority, SyncConfig
from valtexa_rover.hal.interfaces import HttpResponse, NetworkUnavailableError
from valtexa_rover.runtime.client import DeviceClient
from valtexa_rover.runtime.events import EventEngine, EventType
from valtexa_rover.runtime.store import LocalStore

RETRY_STATUS = {401, 408, 425, 429}


class SyncBlockedError(RuntimeError):
    """A nuvem respondeu 403: o robô para de sincronizar até alguém intervir."""


@dataclass(slots=True)
class SyncReport:
    requests: int = 0
    records_sent: int = 0
    files_completed: int = 0
    chunks_sent: int = 0
    bytes_sent: int = 0
    telemetry_sent: int = 0
    failed: int = 0
    offline: bool = False
    blocked: bool = False
    errors: list[str] = field(default_factory=list)

    def merge(self, other: SyncReport) -> None:
        for nome in ("requests", "records_sent", "files_completed", "chunks_sent", "bytes_sent",
                     "telemetry_sent", "failed"):
            setattr(self, nome, getattr(self, nome) + getattr(other, nome))
        self.offline = self.offline or other.offline
        self.blocked = self.blocked or other.blocked
        self.errors += other.errors


class _OfflineError(Exception):
    pass


class SyncEngine:
    def __init__(
        self,
        store: LocalStore,
        client: DeviceClient,
        events: EventEngine,
        clock: Clock,
        config: SyncConfig,
        *,
        versions: Callable[[], dict] | None = None,
        seed: int | None = None,
        orcamento=None,
    ) -> None:
        self.store = store
        self.client = client
        self.events = events
        self.clock = clock
        self.config = config
        self.versions = versions or (lambda: {})
        self._rng = random.Random(seed)
        self.orcamento = orcamento
        """:class:`~valtexa_rover.runtime.orcamento.OrcamentoDados` (``None``: sem teto)."""

    # -- estado -----------------------------------------------------------------

    def pending(self) -> int:
        """Itens que ainda vão subir (inclui telemetria não enviada)."""
        fila = self.store.scalar(
            "SELECT COUNT(*) FROM sync_queue WHERE status IN ('PENDENTE', 'ENVIANDO')"
        )
        tele = self.store.scalar("SELECT COUNT(*) FROM telemetry WHERE sent = 0")
        return int(fila or 0) + int(tele or 0)

    def queue_status(self) -> dict[str, int]:
        return {
            linha["status"]: linha["n"]
            for linha in self.store.query(
                "SELECT status, COUNT(*) AS n FROM sync_queue GROUP BY status"
            )
        }

    def saude(self) -> dict:
        """A fila vista pelo Health Monitor: quanto falta, há quanto tempo e se travou.

        ``oldest_pending_s`` é a idade do item mais antigo ainda na fila (pelo relógio
        de parede gravado na criação): fila que só cresce com rede é o sinal de
        "sincronização parada" que a plataforma alerta.
        """
        antigo = self.store.scalar(
            "SELECT MIN(created_at) FROM sync_queue WHERE status IN ('PENDENTE', 'ENVIANDO')"
        )
        idade = None
        if antigo:
            with contextlib.suppress(ValueError, TypeError):
                idade = max(0.0, (self.clock.now() - datetime.fromisoformat(antigo))
                            .total_seconds())
        return {
            "pending": self.pending(),
            "failed": int(self.store.scalar(
                "SELECT COUNT(*) FROM sync_queue WHERE status = 'FALHOU'") or 0),
            "oldest_pending_s": None if idade is None else round(idade, 1),
            "blocked": self.blocked,
            "last_ok_at": self.store.get("sync_last_ok_at"),
            # orçamento de dados: o que espera o teto e o que ficou retido no disco
            "retained": int(self.store.scalar(
                "SELECT COUNT(*) FROM sync_queue WHERE status = 'RETIDO'") or 0),
            "deferred": self.orcamento.adiados() if self.orcamento is not None else 0,
        }

    @property
    def blocked(self) -> bool:
        return bool(self.store.get("sync_blocked", False))

    # -- laço -------------------------------------------------------------------

    def step(self, max_requests: int = 50) -> SyncReport:
        """Um passo de sincronização, limitado em requisições (não trava a captura)."""
        rel = SyncReport()
        if self.blocked:
            rel.blocked = True
            return rel
        if not self.client.network.is_online():
            self._link(False)
            rel.offline = True
            return rel
        try:
            self._link(True)
            self._announce(rel)
            while rel.requests < max_requests:
                item = self._next_item()
                if item is None:
                    if not self._send_telemetry(rel):
                        # Fila vazia com rede: é o "sincronizado em" que a saúde mostra.
                        self.store.put("sync_last_ok_at", self.clock.now().isoformat())
                        break
                    continue
                if item["kind"] == "record":
                    self._send_records(item, rel)
                else:
                    self._send_file(item, rel, max_requests)
        except _OfflineError:
            rel.offline = True
            self._link(False)
        except SyncBlockedError as exc:
            rel.blocked = True
            rel.errors.append(str(exc))
            self.store.put("sync_blocked", True)
        return rel

    def run_until_idle(self, max_steps: int = 1000) -> SyncReport:
        """Sincroniza até esvaziar a fila (ou cair a rede). Para testes e para o fim da missão."""
        total = SyncReport()
        for _ in range(max_steps):
            rel = self.step()
            total.merge(rel)
            if rel.offline or rel.blocked or rel.requests == 0:
                espera = self._next_due_in()
                if rel.offline or rel.blocked or espera is None:
                    break
                self.clock.sleep(espera)
        return total

    # -- seleção ----------------------------------------------------------------

    def _filtro_orcamento(self) -> tuple[str, tuple]:
        """O pedaço do ``WHERE`` que o orçamento impõe agora (vazio sem orçamento)."""
        if self.orcamento is None:
            return "", ()
        missao = self.orcamento.missao_ativa
        limite = self.orcamento.prioridade_maxima(missao)
        meta = 1 if self.orcamento.metadados_liberados(missao) else 0
        return (" AND (q.priority <= 1 OR (q.kind = 'file' AND q.priority <= ?) OR "
                "(q.kind = 'record' AND (COALESCE(r.kind, '') = 'event' OR ? = 1)))",
                (limite, meta))

    def _next_item(self):
        agora = self.clock.monotonic()
        filtro, params = self._filtro_orcamento()
        linhas = self.store.query(
            "SELECT q.* FROM sync_queue q LEFT JOIN records r "
            "ON q.kind = 'record' AND r.id = q.ref_id "
            "WHERE q.status IN ('PENDENTE', 'ENVIANDO') "
            f"AND q.next_attempt_at <= ?{filtro} ORDER BY q.priority, q.seq LIMIT 1",
            (agora, *params),
        )
        return linhas[0] if linhas else None

    def _next_due_in(self) -> float | None:
        filtro, params = self._filtro_orcamento()
        proximo = self.store.scalar(
            "SELECT MIN(q.next_attempt_at) FROM sync_queue q LEFT JOIN records r "
            "ON q.kind = 'record' AND r.id = q.ref_id "
            f"WHERE q.status IN ('PENDENTE', 'ENVIANDO'){filtro}", params
        )
        if proximo is None:
            return None
        return max(0.0, float(proximo) - self.clock.monotonic())

    # -- registros --------------------------------------------------------------

    def _send_records(self, primeiro, rel: SyncReport) -> None:
        agora = self.clock.monotonic()
        filtro, params = self._filtro_orcamento()
        itens = self.store.query(
            "SELECT q.id AS qid, q.attempts, r.id, r.kind, r.type, r.sha256, r.payload, "
            "r.mission_id FROM sync_queue q JOIN records r ON r.id = q.ref_id "
            "WHERE q.kind = 'record' AND q.status IN ('PENDENTE', 'ENVIANDO') "
            f"AND q.priority = ? AND q.next_attempt_at <= ?{filtro} ORDER BY q.seq LIMIT ?",
            (primeiro["priority"], agora, *params, self.config.batch_size),
        )
        if not itens:
            return
        with self._atribuido(itens[0]["mission_id"]):
            self._send_records_lote(itens, rel)

    def _atribuido(self, mission_id: str | None):
        if self.orcamento is None:
            return contextlib.nullcontext()
        return self.orcamento.atribuir(mission_id)

    def _send_records_lote(self, itens, rel: SyncReport) -> None:
        corpo = {
            "records": [
                {"id": i["id"], "kind": i["kind"], "type": i["type"], "sha256": i["sha256"],
                 "body": json.loads(i["payload"])}
                for i in itens
            ],
            "remaining": max(0, self.pending() - len(itens)),
        }
        resposta = self._request("POST", "/v1/device/edge/records", rel, json_body=corpo,
                                 gzip_body=self.config.comprimir_registros)
        por_qid = {i["id"]: i for i in itens}
        if resposta.status != 200:
            self._falha_ou_retry([i["qid"] for i in itens], [i["attempts"] for i in itens],
                                 resposta, rel, "registros")
            return
        dados = resposta.json()
        ok = set(dados.get("accepted", [])) | set(dados.get("duplicated", []))
        recusados = {r["id"]: r.get("error", "recusado") for r in dados.get("rejected", [])}
        for rid in dados.get("conflicts", []):
            recusados[rid] = "conflito: mesmo ID com conteúdo diferente"
        with self.store.transaction() as conn:
            for rid in ok:
                if rid in por_qid:
                    conn.execute("UPDATE sync_queue SET status = 'CONCLUIDO' WHERE id = ?",
                                 (por_qid[rid]["qid"],))
            for rid, erro in recusados.items():
                if rid in por_qid:
                    conn.execute(
                        "UPDATE sync_queue SET status = 'FALHOU', last_error = ? WHERE id = ?",
                        (erro, por_qid[rid]["qid"]),
                    )
        rel.records_sent += len(ok & set(por_qid))
        if recusados:
            rel.failed += len(recusados)
            rel.errors += [f"{k}: {v}" for k, v in recusados.items()]

    # -- arquivos ---------------------------------------------------------------

    def _send_file(self, item, rel: SyncReport, max_requests: int) -> None:
        missao = self.store.scalar(
            "SELECT r.mission_id FROM files f JOIN records r ON r.id = f.record_id "
            "WHERE f.id = ?", (item["ref_id"],))
        with self._atribuido(missao):
            self._send_file_item(item, rel, max_requests)

    def _send_file_item(self, item, rel: SyncReport, max_requests: int) -> None:
        linhas = self.store.query("SELECT * FROM files WHERE id = ?", (item["ref_id"],))
        if not linhas:
            self._marca(item["id"], "FALHOU", "arquivo removido pela retenção")
            rel.failed += 1
            return
        arquivo = linhas[0]
        caminho = self.store.blob_path(arquivo["path"])
        if not caminho.exists():
            self._marca(item["id"], "FALHOU", "arquivo local ausente")
            rel.failed += 1
            return
        abrir = {
            "upload_id": item["id"],
            "sha256": arquivo["sha256"],
            "size_bytes": arquivo["size_bytes"],
            "content_type": arquivo["content_type"],
            "chunk_size": item["chunk_size"] or self.config.chunk_size,
            "record_id": arquivo["record_id"],
            "role": arquivo["role"],
            "tier": arquivo["tier"],
            "priority": item["priority"],
        }
        primeira_vez = item["status"] == "PENDENTE" and item["chunks_done"] == 0
        resposta = self._request("POST", "/v1/device/uploads", rel, json_body=abrir)
        if resposta.status not in (200, 201):
            self._falha_ou_retry([item["id"]], [item["attempts"]], resposta, rel, "upload")
            return
        estado = resposta.json()
        total = int(estado["total_chunks"])
        recebidos = set(estado.get("received_chunks", []))
        tamanho_chunk = int(estado["chunk_size"])
        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE sync_queue SET status = 'ENVIANDO', chunk_size = ?, chunks_total = ?, "
                "chunks_done = ? WHERE id = ?",
                (tamanho_chunk, total, len(recebidos), item["id"]),
            )
        ja_tinha = estado.get("status") == "complete"
        if primeira_vez and not ja_tinha and item["priority"] <= Priority.ANOMALIA:
            self.events.emit(EventType.UPLOAD_STARTED,
                             {"upload_id": item["id"], "sha256": arquivo["sha256"],
                              "size_bytes": arquivo["size_bytes"], "role": arquivo["role"]})

        if not ja_tinha:
            with open(caminho, "rb") as fh:
                for indice in range(total):
                    if indice in recebidos:
                        continue
                    if rel.requests >= max_requests:
                        return  # continua no próximo passo, do chunk em que parou
                    fh.seek(indice * tamanho_chunk)
                    pedaco = fh.read(tamanho_chunk)
                    r = self._request(
                        "PUT", f"/v1/device/uploads/{item['id']}/chunks/{indice}", rel,
                        raw=pedaco, content_type="application/octet-stream",
                    )
                    if r.status != 200:
                        self._falha_ou_retry([item["id"]], [item["attempts"]], r, rel, "chunk")
                        return
                    recebidos.add(indice)
                    rel.chunks_sent += 1
                    rel.bytes_sent += len(pedaco)
                    with self.store.transaction() as conn:
                        conn.execute("UPDATE sync_queue SET chunks_done = ? WHERE id = ?",
                                     (len(recebidos), item["id"]))
            r = self._request("POST", f"/v1/device/uploads/{item['id']}/complete", rel,
                              json_body={"sha256": arquivo["sha256"]})
            if r.status != 200:
                self._falha_ou_retry([item["id"]], [item["attempts"]], r, rel, "conclusão")
                return
        with self.store.transaction() as conn:
            conn.execute("UPDATE sync_queue SET status = 'CONCLUIDO', chunks_done = chunks_total "
                         "WHERE id = ?", (item["id"],))
            conn.execute("UPDATE files SET uploaded = 1 WHERE id = ?", (arquivo["id"],))
        rel.files_completed += 1
        if not ja_tinha and item["priority"] <= Priority.ANOMALIA:
            self.events.emit(EventType.UPLOAD_COMPLETED,
                             {"upload_id": item["id"], "sha256": arquivo["sha256"]})

    # -- telemetria e versões ---------------------------------------------------

    def _send_telemetry(self, rel: SyncReport) -> bool:
        linhas = self.store.query(
            "SELECT sequence_number, payload FROM telemetry WHERE sent = 0 "
            "ORDER BY sequence_number LIMIT ?",
            (min(self.config.batch_size, 500),),
        )
        if not linhas:
            return False
        amostras = [json.loads(linha["payload"]) for linha in linhas]
        primeiro, ultimo = linhas[0]["sequence_number"], linhas[-1]["sequence_number"]
        corpo = {
            "sync_id": f"tele-{primeiro}-{ultimo}",
            "samples": amostras,
            "events": [],
            "remaining": max(0, self.pending() - len(linhas)),
        }
        resposta = self._request("POST", "/v1/device/sync", rel, json_body=corpo)
        if resposta.status != 200:
            if resposta.status == 403:
                raise SyncBlockedError("403 na telemetria")
            rel.errors.append(f"telemetria: HTTP {resposta.status}")
            return False
        with self.store.transaction() as conn:
            conn.executemany("UPDATE telemetry SET sent = 1 WHERE sequence_number = ?",
                             [(linha["sequence_number"],) for linha in linhas])
        rel.telemetry_sent += len(linhas)
        return True

    def _announce(self, rel: SyncReport) -> None:
        """Informa as versões (runtime, Edge AI, modelo, dNATY, hardware) quando mudam."""
        versoes = {"runtime_version": RUNTIME_VERSION, "edge_ai_version": EDGE_AI_VERSION,
                   **{k: v for k, v in self.versions().items() if v is not None}}
        if self.store.get("versions_reported") == versoes:
            return
        resposta = self._request("POST", "/v1/device/session", rel, json_body=versoes)
        if resposta.status == 200:
            self.store.put("versions_reported", versoes)
        elif resposta.status == 403:
            raise SyncBlockedError("403 ao abrir sessão")

    # -- utilidades -------------------------------------------------------------

    def _request(self, method: str, path: str, rel: SyncReport, **kwargs) -> HttpResponse:
        try:
            resposta = self.client.request(method, path, **kwargs)
        except NetworkUnavailableError as exc:
            raise _OfflineError() from exc
        rel.requests += 1
        if resposta.status == 403:
            raise SyncBlockedError(f"403 em {method} {path}: {resposta.body[:200]!r}")
        return resposta

    def _falha_ou_retry(
        self, qids: list[str], tentativas: list[int], resposta: HttpResponse,
        rel: SyncReport, oque: str,
    ) -> None:
        status = resposta.status
        detalhe = resposta.body[:300].decode("utf-8", "replace")
        definitivo = 400 <= status < 500 and status not in RETRY_STATUS
        with self.store.transaction() as conn:
            for qid, feitas in zip(qids, tentativas, strict=True):
                feitas += 1
                esgotou = self.config.max_attempts and feitas >= self.config.max_attempts
                if definitivo or esgotou:
                    conn.execute(
                        "UPDATE sync_queue SET status = 'FALHOU', attempts = ?, last_error = ? "
                        "WHERE id = ?",
                        (feitas, f"HTTP {status}: {detalhe}", qid),
                    )
                else:
                    espera = min(self.config.backoff_max_s,
                                 self.config.backoff_base_s * 2 ** (feitas - 1))
                    espera *= 0.5 + self._rng.random() / 2  # jitter
                    conn.execute(
                        "UPDATE sync_queue SET attempts = ?, next_attempt_at = ?, last_error = ? "
                        "WHERE id = ?",
                        (feitas, self.clock.monotonic() + espera, f"HTTP {status}: {detalhe}",
                         qid),
                    )
        rel.errors.append(f"{oque}: HTTP {status}")
        if definitivo:
            rel.failed += len(qids)
            for qid in qids:
                self.events.emit(EventType.UPLOAD_FAILED,
                                 {"queue_id": qid, "status": status, "detail": detalhe})

    def _marca(self, qid: str, status: str, erro: str) -> None:
        with self.store.transaction() as conn:
            conn.execute("UPDATE sync_queue SET status = ?, last_error = ? WHERE id = ?",
                         (status, erro, qid))

    def _link(self, online: bool) -> None:
        """Registra a troca de estado do enlace (persistida: vale entre reinícios)."""
        anterior = self.store.get("link_online")
        if anterior is not None and online != anterior:
            self.events.emit(EventType.ROVER_ONLINE if online else EventType.ROVER_OFFLINE,
                             {"pending": self.pending()})
        if anterior != online:
            self.store.put("link_online", online)
