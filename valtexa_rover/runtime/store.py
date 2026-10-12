"""Local Storage: SQLite transacional + arquivos endereçados por hash.

Brief, seção 12: sem rede o Rover continua, e **nada se perde num reinício no
meio da missão**. Como isso é garantido:

* o banco é SQLite em modo WAL com ``synchronous=FULL``: um *commit* que voltou
  está no disco, mesmo com o robô desligado na tomada logo depois;
* tudo o que um quadro produz (resultado, evento, arquivos, itens da fila de
  sincronização, progresso da missão) entra **numa transação só**
  (:meth:`LocalStore.transaction`): ou o quadro inteiro existe, ou nada dele;
* o arquivo é gravado **antes** do *commit*, num ``.part`` com ``fsync`` e
  renomeado (operação atômica). Se o robô cair entre o arquivo e o *commit*,
  sobra um arquivo sem linha — :meth:`LocalStore.recover` o apaga no boot. O
  contrário (linha sem arquivo) não acontece;
* os arquivos ficam em ``files/ab/cd/<sha256><ext>``: o mesmo conteúdo é gravado
  uma vez, e o hash que vai na fila é o do arquivo que está no disco.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS missions (
    id TEXT PRIMARY KEY,
    field_id TEXT,
    state TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    frames_done INTEGER NOT NULL DEFAULT 0,
    events INTEGER NOT NULL DEFAULT 0,
    model_key TEXT,
    model_version TEXT,
    model_sha256 TEXT,
    payload TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS records (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,           -- result | event
    type TEXT NOT NULL,           -- categoria do resultado ou tipo do evento
    mission_id TEXT,
    occurred_at TEXT NOT NULL,
    sha256 TEXT NOT NULL,         -- do JSON canônico de payload: o que a nuvem confere
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_records_mission ON records (mission_id, kind);
CREATE TABLE IF NOT EXISTS files (
    id TEXT PRIMARY KEY,
    record_id TEXT NOT NULL,
    role TEXT NOT NULL,           -- original | reduzida | contexto
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    content_type TEXT NOT NULL,
    tier TEXT NOT NULL,
    retention_until TEXT NOT NULL,
    uploaded INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_files_record ON files (record_id);
CREATE INDEX IF NOT EXISTS ix_files_path ON files (path);
CREATE TABLE IF NOT EXISTS sync_queue (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,           -- record | file
    ref_id TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    chunk_size INTEGER NOT NULL DEFAULT 0,
    chunks_total INTEGER NOT NULL DEFAULT 0,
    chunks_done INTEGER NOT NULL DEFAULT 0,
    priority INTEGER NOT NULL,
    status TEXT NOT NULL,         -- PENDENTE | ENVIANDO | CONCLUIDO | FALHOU
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at REAL NOT NULL DEFAULT 0,
    last_error TEXT,
    created_at TEXT NOT NULL,
    seq INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_queue_ready ON sync_queue (status, priority, seq);
CREATE TABLE IF NOT EXISTS telemetry (
    sequence_number INTEGER PRIMARY KEY,
    recorded_at TEXT NOT NULL,
    payload TEXT NOT NULL,
    sent INTEGER NOT NULL DEFAULT 0
);
"""


def canonical_json(payload: object) -> bytes:
    """JSON canônico (chaves ordenadas, sem espaços): o hash de um registro sai daqui."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class LocalStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.files_dir = self.root / "files"
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "rover.db"
        self._lock = threading.RLock()
        self._depth = 0
        self.conn = sqlite3.connect(
            self.db_path, isolation_level=None, check_same_thread=False, timeout=30
        )
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        if self.get("schema_version") is None:
            self.put("schema_version", SCHEMA_VERSION)

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    # -- transações -----------------------------------------------------------

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """``BEGIN IMMEDIATE`` … ``COMMIT``; exceção → ``ROLLBACK``. Reentrante."""
        with self._lock:
            if self._depth:
                self._depth += 1
                try:
                    yield self.conn
                finally:
                    self._depth -= 1
                return
            self.conn.execute("BEGIN IMMEDIATE")
            self._depth = 1
            try:
                yield self.conn
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            else:
                self.conn.execute("COMMIT")
            finally:
                self._depth = 0

    def query(self, sql: str, params: tuple | list = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self.conn.execute(sql, params))

    def scalar(self, sql: str, params: tuple | list = ()):
        linhas = self.query(sql, params)
        return linhas[0][0] if linhas else None

    # -- chave/valor ----------------------------------------------------------

    def get(self, key: str, default=None):
        valor = self.scalar("SELECT value FROM kv WHERE key = ?", (key,))
        return default if valor is None else json.loads(valor)

    def put(self, key: str, value) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO kv (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, json.dumps(value)),
            )

    def next_counter(self, name: str) -> int:
        """Contador monotônico persistido (sequência do protocolo, ordem da fila)."""
        with self.transaction():
            valor = int(self.get(name, 0)) + 1
            self.put(name, valor)
            return valor

    # -- arquivos -------------------------------------------------------------

    def write_blob(self, data: bytes, suffix: str) -> tuple[str, str]:
        """Grava o conteúdo (atômico, com ``fsync``) e devolve ``(caminho relativo, sha256)``."""
        digest = sha256_hex(data)
        relativo = f"files/{digest[:2]}/{digest[2:4]}/{digest}{suffix}"
        destino = self.root / relativo
        if destino.exists():
            return relativo, digest
        destino.parent.mkdir(parents=True, exist_ok=True)
        temporario = destino.with_name(destino.name + ".part")
        with open(temporario, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporario, destino)
        return relativo, digest

    def blob_path(self, relativo: str) -> Path:
        return self.root / relativo

    def delete_blob_if_unreferenced(self, relativo: str) -> bool:
        if self.scalar("SELECT COUNT(*) FROM files WHERE path = ?", (relativo,)):
            return False
        caminho = self.root / relativo
        if caminho.exists():
            caminho.unlink()
            return True
        return False

    def recover(self) -> dict[str, int]:
        """No boot: apaga ``.part`` e arquivos sem linha; marca linhas cujo arquivo sumiu."""
        referenciados = {row[0] for row in self.query("SELECT DISTINCT path FROM files")}
        orfaos = parciais = 0
        for caminho in self.files_dir.rglob("*"):
            if not caminho.is_file():
                continue
            relativo = caminho.relative_to(self.root).as_posix()
            if caminho.name.endswith(".part"):
                caminho.unlink()
                parciais += 1
            elif relativo not in referenciados:
                caminho.unlink()
                orfaos += 1
        perdidos = 0
        for relativo in referenciados:
            if not (self.root / relativo).exists():
                perdidos += 1
                with self.transaction() as conn:
                    conn.execute(
                        "UPDATE sync_queue SET status = 'FALHOU', last_error = ? "
                        "WHERE kind = 'file' AND ref_id IN (SELECT id FROM files WHERE path = ?) "
                        "AND status != 'CONCLUIDO'",
                        ("arquivo local ausente", relativo),
                    )
        with self.transaction() as conn:
            # Um item que estava "ENVIANDO" quando o robô caiu volta à fila: o
            # progresso dele (chunks_done) foi gravado chunk a chunk e continua valendo.
            conn.execute("UPDATE sync_queue SET status = 'PENDENTE' WHERE status = 'ENVIANDO'")
            # ``next_attempt_at`` está no relógio monotônico, que recomeça do zero a
            # cada boot: um *backoff* gravado antes do reinício prenderia o item por
            # horas. O reinício é um bom momento para tentar de novo.
            conn.execute(
                "UPDATE sync_queue SET next_attempt_at = 0 WHERE status = 'PENDENTE'"
            )
        return {"parciais": parciais, "orfaos": orfaos, "perdidos": perdidos}

    def disk_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.files_dir.rglob("*") if p.is_file())


def iso(dt: datetime) -> str:
    return dt.isoformat()
