"""Filas persistentes do enlace (SQLite em WAL): queda de energia não perde nada.

Três tabelas, nas duas pontas:

* ``saida`` — o que este nó tem de entregar. A mensagem entra **antes** de
  :meth:`FilaPersistente.enfileira` voltar e só sai quando o outro lado confirma que a
  guardou ("entregue inteira"). Depois de um reinício tudo é reenviado; o outro lado
  descarta o repetido pela tabela ``vistos``.
* ``entrada`` — o que chegou inteiro e ainda não foi consumido pela aplicação. Entra na
  mesma transação que o ``vistos``: ou as duas coisas, ou nenhuma.
* ``vistos`` — ids já entregues à ``entrada`` (deduplicação; guardados por
  ``retencao_vistos_s``).

**Limites e disco cheio** (P0 a P3 nunca caem por causa de P4):

* P4 tem teto próprio (``limite_p4_bytes``); acima dele o P4 novo é recusado
  (:class:`FilaCheiaError` — quem chamou guarda e tenta depois; no Rover o dado continua na
  fila de sincronização, que só esvazia com o aceite da plataforma);
* P2+P3+P4 juntos têm teto (``limite_bytes``): para caber P2/P3, o P4 mais velho da fila
  é **removido da fila do enlace** (a origem dele continua no disco de quem pediu);
  se ainda não couber, o P2/P3 novo é recusado — nunca se apaga P0-P3 já aceito;
* P0/P1 têm reserva própria (``reserva_p01_bytes``) e não disputam com ninguém;
* com menos de ``minimo_livre_bytes`` livres no disco, só P0/P1 entram e o P4 sai.

Mensagem com validade (``expira``, hora de parede) que vence na fila é descartada, nunca
enviada; a contagem fica em ``descartes``.
"""

from __future__ import annotations

import shutil
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ESQUEMA = """
CREATE TABLE IF NOT EXISTS saida (
    id BLOB PRIMARY KEY, par TEXT NOT NULL, prio INTEGER NOT NULL, canal INTEGER NOT NULL,
    corpo BLOB NOT NULL, criado REAL NOT NULL, expira REAL, tamanho INTEGER NOT NULL,
    ordem INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS saida_ordem ON saida (par, prio, ordem);
CREATE TABLE IF NOT EXISTS entrada (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, id BLOB NOT NULL, par TEXT NOT NULL,
    prio INTEGER NOT NULL, canal INTEGER NOT NULL, corpo BLOB NOT NULL,
    recebido REAL NOT NULL, expira REAL
);
CREATE INDEX IF NOT EXISTS entrada_canal ON entrada (canal, seq);
CREATE TABLE IF NOT EXISTS vistos (
    par TEXT NOT NULL, id BLOB NOT NULL, quando REAL NOT NULL, PRIMARY KEY (par, id)
);
CREATE TABLE IF NOT EXISTS contadores (nome TEXT PRIMARY KEY, valor INTEGER NOT NULL);
"""


class FilaCheiaError(RuntimeError):
    """Não coube: quem chamou guarda o dado e tenta depois (nada é apagado em silêncio)."""


class DiscoCheioError(FilaCheiaError):
    """O SQLite não conseguiu gravar (disco cheio de verdade, ou com defeito)."""


@dataclass(frozen=True, slots=True)
class Saindo:
    id: bytes
    par: str
    prio: int
    canal: int
    corpo: bytes
    criado: float
    expira: float | None


@dataclass(frozen=True, slots=True)
class Chegada:
    seq: int
    id: bytes
    par: str
    prio: int
    canal: int
    corpo: bytes
    recebido: float
    expira: float | None


@dataclass
class Limites:
    limite_bytes: int = 512 * 1024 * 1024
    limite_p4_bytes: int = 384 * 1024 * 1024
    reserva_p01_bytes: int = 32 * 1024 * 1024
    minimo_livre_bytes: int = 256 * 1024 * 1024
    retencao_vistos_s: float = 7 * 24 * 3600


class FilaPersistente:
    def __init__(self, caminho: Path | str, parede: Callable[[], float], *,
                 limites: Limites | None = None,
                 livre_no_disco: Callable[[], int] | None = None,
                 sincrono: str = "FULL") -> None:
        """``sincrono="FULL"`` (produção): cada commit vai ao disco antes de voltar — sobrevive
        à queda de energia. ``"NORMAL"`` (testes): sobrevive ao ``kill -9`` do processo, não à
        queda de energia; o caos em relógio virtual não precisa pagar um ``fsync`` por
        mensagem."""
        self.caminho = Path(caminho)
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        self.parede = parede
        self.limites = limites or Limites()
        self._livre = livre_no_disco or (lambda: shutil.disk_usage(self.caminho.parent).free)
        self._trava = threading.RLock()
        self._db = sqlite3.connect(self.caminho, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        if sincrono not in ("FULL", "NORMAL"):
            raise ValueError("sincrono: FULL ou NORMAL")
        self._db.execute(f"PRAGMA synchronous={sincrono}")
        self._db.executescript(ESQUEMA)
        self._ordem = (self._db.execute("SELECT COALESCE(MAX(ordem), 0) FROM saida")
                       .fetchone()[0])
        self.descartes = {"expirada": 0, "p4_removida": 0, "recusada": 0}

    def fechar(self) -> None:
        with self._trava:
            self._db.close()

    # -- saída ------------------------------------------------------------------

    def _bytes(self, onde: str, args: tuple = ()) -> int:
        return self._db.execute(f"SELECT COALESCE(SUM(tamanho), 0) FROM saida WHERE {onde}",
                                args).fetchone()[0]

    def enfileira(self, msg_id: bytes, par: str, prio: int, canal: int, corpo: bytes,
                  expira: float | None) -> None:
        n = len(corpo)
        lim = self.limites
        with self._trava:
            disco_baixo = self._livre() < lim.minimo_livre_bytes
            if prio <= 1:
                if self._bytes("prio <= 1") + n > lim.reserva_p01_bytes:
                    self.descartes["recusada"] += 1
                    raise FilaCheiaError("reserva de P0/P1 cheia")
            else:
                if disco_baixo:
                    self._remove_p4(None)
                    self.descartes["recusada"] += 1
                    raise FilaCheiaError("disco quase cheio: só P0/P1 entram")
                if prio == 4 and self._bytes("prio = 4") + n > lim.limite_p4_bytes:
                    self.descartes["recusada"] += 1
                    raise FilaCheiaError("teto de P4 atingido")
                excesso = self._bytes("prio >= 2") + n - lim.limite_bytes
                if excesso > 0 and prio < 4 and self._bytes("prio = 4") >= excesso:
                    excesso -= self._remove_p4(excesso)  # só tira P4 se isso resolver
                if excesso > 0:
                    self.descartes["recusada"] += 1
                    raise FilaCheiaError("fila do enlace cheia")
            self._ordem += 1
            try:
                self._db.execute(
                    "INSERT INTO saida (id, par, prio, canal, corpo, criado, expira, tamanho,"
                    " ordem) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (msg_id, par, prio, canal, corpo, self.parede(), expira, n, self._ordem))
            except sqlite3.OperationalError as exc:  # SQLITE_FULL / IOERR
                self.descartes["recusada"] += 1
                raise DiscoCheioError(f"não gravou na fila do enlace: {exc}") from exc

    def _remove_p4(self, bytes_a_liberar: int | None) -> int:
        """Tira P4 da fila (mais velho primeiro). ``None`` = todo o P4."""
        liberado = 0
        for linha in self._db.execute(
                "SELECT id, tamanho FROM saida WHERE prio = 4 ORDER BY ordem").fetchall():
            if bytes_a_liberar is not None and liberado >= bytes_a_liberar:
                break
            self._db.execute("DELETE FROM saida WHERE id = ?", (linha["id"],))
            liberado += linha["tamanho"]
            self.descartes["p4_removida"] += 1
        return liberado

    def pendentes(self, par: str | None = None, prio: int | None = None,
                  limite: int | None = None, excluir: set[bytes] | frozenset = frozenset()
                  ) -> list[Saindo]:
        """O que falta entregar, na ordem (prioridade, chegada). Vencidas são descartadas.

        ``excluir``: ids que o nó já tem em voo (não carrega o corpo de novo).
        """
        agora = self.parede()
        with self._trava:
            vencidas = self._db.execute(
                "DELETE FROM saida WHERE expira IS NOT NULL AND expira <= ?", (agora,)).rowcount
            self.descartes["expirada"] += vencidas
            onde, args = [], []
            if par is not None:
                onde.append("par = ?")
                args.append(par)
            if prio is not None:
                onde.append("prio = ?")
                args.append(prio)
            sql = "SELECT * FROM saida" + (" WHERE " + " AND ".join(onde) if onde else "")
            sql += " ORDER BY prio, ordem"
            saida: list[Saindo] = []
            for r in self._db.execute(sql, args):
                if r["id"] in excluir:
                    continue
                saida.append(Saindo(r["id"], r["par"], r["prio"], r["canal"], r["corpo"],
                                    r["criado"], r["expira"]))
                if limite is not None and len(saida) >= limite:
                    break
        return saida

    def ids_pendentes(self) -> set[bytes]:
        with self._trava:
            return {r[0] for r in self._db.execute("SELECT id FROM saida")}

    def entregue(self, msg_id: bytes) -> bool:
        with self._trava:
            return self._db.execute("DELETE FROM saida WHERE id = ?", (msg_id,)).rowcount > 0

    def descarta(self, msg_id: bytes) -> None:
        """Validade vencida durante o envio."""
        with self._trava:
            if self._db.execute("DELETE FROM saida WHERE id = ?", (msg_id,)).rowcount:
                self.descartes["expirada"] += 1

    def tamanho_saida(self) -> dict[int, int]:
        with self._trava:
            return {r[0]: r[1] for r in self._db.execute(
                "SELECT prio, COUNT(*) FROM saida GROUP BY prio")}

    # -- entrada ----------------------------------------------------------------

    def ja_visto(self, par: str, msg_id: bytes) -> bool:
        with self._trava:
            return self._db.execute("SELECT 1 FROM vistos WHERE par = ? AND id = ?",
                                    (par, msg_id)).fetchone() is not None

    def recebe(self, par: str, msg_id: bytes, prio: int, canal: int, corpo: bytes,
               expira: float | None) -> bool:
        """Guarda a mensagem inteira. ``False`` se já tinha (repetida: não entra de novo)."""
        agora = self.parede()
        with self._trava:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                novo = self._db.execute(
                    "INSERT OR IGNORE INTO vistos (par, id, quando) VALUES (?, ?, ?)",
                    (par, msg_id, agora)).rowcount > 0
                if novo:
                    self._db.execute(
                        "INSERT INTO entrada (id, par, prio, canal, corpo, recebido, expira) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (msg_id, par, prio, canal, corpo, agora, expira))
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
        return novo

    def chegadas(self, canal: int | None = None, limite: int = 100,
                 excluir_canal: int | None = None) -> list[Chegada]:
        with self._trava:
            sql, args = "SELECT * FROM entrada", ()
            if canal is not None:
                sql, args = sql + " WHERE canal = ?", (canal,)
            elif excluir_canal is not None:
                sql, args = sql + " WHERE canal != ?", (excluir_canal,)
            linhas = self._db.execute(sql + " ORDER BY prio, seq LIMIT ?",
                                      (*args, limite)).fetchall()
        return [Chegada(r["seq"], r["id"], r["par"], r["prio"], r["canal"], r["corpo"],
                        r["recebido"], r["expira"]) for r in linhas]

    def consumida(self, seq: int) -> None:
        with self._trava:
            self._db.execute("DELETE FROM entrada WHERE seq = ?", (seq,))

    def limpa_entrada_vencida(self, folga_s: float = 60.0) -> int:
        """Chegada com validade vencida que ninguém consumiu (a resposta de quem já desistiu
        de esperar) não fica na ``entrada`` para sempre."""
        with self._trava:
            return self._db.execute(
                "DELETE FROM entrada WHERE expira IS NOT NULL AND expira < ?",
                (self.parede() - folga_s,)).rowcount

    def tamanho_entrada(self) -> int:
        with self._trava:
            return self._db.execute("SELECT COUNT(*) FROM entrada").fetchone()[0]

    def limpa_vistos(self) -> int:
        with self._trava:
            return self._db.execute(
                "DELETE FROM vistos WHERE quando < ?",
                (self.parede() - self.limites.retencao_vistos_s,)).rowcount

    # -- contadores persistentes (ex.: sequência de slot) --------------------------

    def contador(self, nome: str, incremento: int = 1) -> int:
        with self._trava:
            self._db.execute(
                "INSERT INTO contadores (nome, valor) VALUES (?, ?) "
                "ON CONFLICT(nome) DO UPDATE SET valor = valor + ?",
                (nome, incremento, incremento))
            return self._db.execute("SELECT valor FROM contadores WHERE nome = ?",
                                    (nome,)).fetchone()[0]
