"""Transferência retomável de arquivo grande pelo enlace, em pedaços com hash.

* O emissor manda um **manifesto** (nome, tamanho, SHA-256 do arquivo, tamanho do pedaço
  e o SHA-256 de cada pedaço) e depois cada **pedaço** como uma mensagem própria do
  enlace (P4 por padrão), numa janela de ``janela`` pedaços enfileirados por vez — o
  arquivo não é copiado inteiro para a fila.
* O progresso do emissor fica num SQLite ao lado: qual pedaço já foi enfileirado e com
  qual id de mensagem. Pedaço entregue = mensagem que saiu da fila do enlace. Depois de um
  reinício, os já entregues não saem de novo; os enfileirados continuam na fila
  persistente; os outros entram quando houver janela.
* O receptor confere o hash de cada pedaço, escreve no lugar dele num ``.part`` (com
  ``fsync``) e guarda o mapa do que já tem; pedaço que chega antes do manifesto espera na
  entrada. Completo, confere o SHA-256 do arquivo e renomeia.

Pedaço repetido (reenvio depois de reinício) é reconhecido pelo mapa e não é escrito de
novo.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path

from valtexa_rover.enlace.no import P3, P4, NoDeEnlace
from valtexa_rover.enlace.rpc import CANAL_ARQUIVO, codifica, decodifica

PECA_PADRAO = 64 * 1024


def _sha(dados: bytes) -> str:
    return hashlib.sha256(dados).hexdigest()


def manifesto_de(caminho: Path, tid: str, peca: int = PECA_PADRAO) -> dict:
    hashes, total = [], hashlib.sha256()
    tamanho = 0
    with open(caminho, "rb") as fh:
        while bloco := fh.read(peca):
            hashes.append(_sha(bloco))
            total.update(bloco)
            tamanho += len(bloco)
    return {"tid": tid, "nome": caminho.name, "tamanho": tamanho, "sha256": total.hexdigest(),
            "peca": peca, "hashes": hashes}


class EnviaArquivos:
    def __init__(self, no: NoDeEnlace, par: str, estado: Path | str, *, janela: int = 8,
                 prio: int = P4) -> None:
        self.no = no
        self.par = par
        self.janela = janela
        self.prio = prio
        self._db = sqlite3.connect(estado, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(
            "CREATE TABLE IF NOT EXISTS transf (tid TEXT PRIMARY KEY, caminho TEXT NOT NULL, "
            "manifesto TEXT NOT NULL, man_msg BLOB);"
            "CREATE TABLE IF NOT EXISTS pecas (tid TEXT NOT NULL, i INTEGER NOT NULL, "
            "msg BLOB NOT NULL, PRIMARY KEY (tid, i));")
        self.pecas_enfileiradas = 0

    def oferece(self, caminho: Path | str, tid: str, peca: int = PECA_PADRAO) -> dict:
        """Registra a transferência (idempotente: o mesmo ``tid`` não recomeça)."""
        linha = self._db.execute("SELECT manifesto FROM transf WHERE tid = ?", (tid,)).fetchone()
        if linha:
            return json.loads(linha[0])
        man = manifesto_de(Path(caminho), tid, peca)
        self._db.execute("INSERT INTO transf (tid, caminho, manifesto) VALUES (?, ?, ?)",
                         (tid, str(caminho), json.dumps(man)))
        return man

    def progresso(self, tid: str) -> tuple[int, int]:
        """``(pedaços entregues, total)``."""
        man = json.loads(self._db.execute("SELECT manifesto FROM transf WHERE tid = ?",
                                          (tid,)).fetchone()[0])
        pendentes = self.no.fila.ids_pendentes()
        entregues = sum(1 for (msg,) in self._db.execute(
            "SELECT msg FROM pecas WHERE tid = ?", (tid,)) if msg not in pendentes)
        return entregues, len(man["hashes"])

    def bombeia(self) -> None:
        """Enfileira manifesto e pedaços até a janela. Chame a cada volta (é barato)."""
        pendentes = self.no.fila.ids_pendentes()
        for tid, caminho, man_txt, man_msg in self._db.execute(
                "SELECT tid, caminho, manifesto, man_msg FROM transf").fetchall():
            man = json.loads(man_txt)
            if man_msg is None:
                mid = self.no.enviar(self.par, codifica({"t": "man", **man}), P3, CANAL_ARQUIVO)
                self._db.execute("UPDATE transf SET man_msg = ? WHERE tid = ?", (mid, tid))
            feitas = dict(self._db.execute("SELECT i, msg FROM pecas WHERE tid = ?", (tid,)))
            em_voo = sum(1 for m in feitas.values() if m in pendentes)
            if len(feitas) == len(man["hashes"]):
                continue
            with open(caminho, "rb") as fh:
                for i in range(len(man["hashes"])):
                    if em_voo >= self.janela:
                        break
                    if i in feitas:
                        continue
                    fh.seek(i * man["peca"])
                    dados = fh.read(man["peca"])
                    mid = self.no.enviar(self.par, codifica(
                        {"t": "peca", "tid": tid, "i": i, "sha": man["hashes"][i]}, dados),
                        self.prio, CANAL_ARQUIVO)
                    self._db.execute("INSERT INTO pecas (tid, i, msg) VALUES (?, ?, ?)",
                                     (tid, i, mid))
                    self.pecas_enfileiradas += 1
                    em_voo += 1

    def fechar(self) -> None:
        self._db.close()


@dataclass
class _Estado:
    man: dict
    tem: set[int]


class RecebeArquivos:
    def __init__(self, no: NoDeEnlace, pasta: Path | str) -> None:
        self.no = no
        self.pasta = Path(pasta)
        self.pasta.mkdir(parents=True, exist_ok=True)
        self.concluidos: list[Path] = []
        self.pecas_escritas = 0
        self.pecas_repetidas = 0
        self.pecas_ruins = 0

    def _estado(self, tid: str) -> _Estado | None:
        f = self.pasta / f"{tid}.estado.json"
        if not f.exists():
            return None
        d = json.loads(f.read_text(encoding="utf-8"))
        return _Estado(d["man"], set(d["tem"]))

    def _grava_estado(self, tid: str, est: _Estado) -> None:
        f = self.pasta / f"{tid}.estado.json"
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps({"man": est.man, "tem": sorted(est.tem)}), encoding="utf-8")
        os.replace(tmp, f)

    def processa(self) -> int:
        n = 0
        for c in self.no.chegadas(CANAL_ARQUIVO, 64):
            try:
                cab, dados = decodifica(c.corpo)
            except (ValueError, struct.error):
                self.no.consumida(c.seq)
                continue
            if cab.get("t") == "man":
                tid = cab["tid"]
                if self._estado(tid) is None and not (self.pasta / cab["nome"]).exists():
                    man = {k: v for k, v in cab.items() if k != "t"}
                    with open(self.pasta / f"{tid}.part", "wb") as fh:
                        fh.truncate(man["tamanho"])
                    self._grava_estado(tid, _Estado(man, set()))
                self.no.consumida(c.seq)
                n += 1
                continue
            tid, i = cab["tid"], int(cab["i"])
            est = self._estado(tid)
            if est is None:
                if any((self.pasta / p).exists() for p in (f"{tid}.ok",)):
                    self.no.consumida(c.seq)  # já concluído: pedaço repetido
                    self.pecas_repetidas += 1
                continue  # o manifesto ainda não chegou: espera na entrada
            self.no.consumida(c.seq)
            n += 1
            if i in est.tem:
                self.pecas_repetidas += 1
                continue
            if _sha(dados) != est.man["hashes"][i] or cab.get("sha") != est.man["hashes"][i]:
                self.pecas_ruins += 1
                continue
            with open(self.pasta / f"{tid}.part", "r+b") as fh:
                fh.seek(i * est.man["peca"])
                fh.write(dados)
                fh.flush()
                os.fsync(fh.fileno())
            est.tem.add(i)
            self.pecas_escritas += 1
            self._grava_estado(tid, est)
            if len(est.tem) == len(est.man["hashes"]):
                self._conclui(tid, est)
        return n

    def _conclui(self, tid: str, est: _Estado) -> None:
        part = self.pasta / f"{tid}.part"
        h = hashlib.sha256()
        with open(part, "rb") as fh:
            while bloco := fh.read(1 << 20):
                h.update(bloco)
        if h.hexdigest() != est.man["sha256"]:
            self.pecas_ruins += 1
            est.tem.clear()
            self._grava_estado(tid, est)
            return
        final = self.pasta / est.man["nome"]
        os.replace(part, final)
        (self.pasta / f"{tid}.ok").write_text(est.man["sha256"], encoding="utf-8")
        (self.pasta / f"{tid}.estado.json").unlink(missing_ok=True)
        self.concluidos.append(final)
