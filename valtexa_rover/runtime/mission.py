"""Mission Manager: a missão vive no banco local e sobrevive a reinício.

Estados: ``EM_EXECUCAO`` → ``CONCLUIDA`` (ou ``INTERROMPIDA`` por decisão). Um
reinício no meio não muda o estado: no boot, :meth:`MissionManager.current`
devolve a missão em execução com o progresso gravado (quadros, eventos), e o
Core continua dela — com ``MISSION_RESUMED``.

A missão chega de dois jeitos:

* da plataforma, como **comando assinado** (``GET /v1/device/missions/current``).
  :func:`verify_command` faz as cinco conferências da seção 6 do protocolo
  (assinatura Ed25519 sobre o JSON canônico, destinatário, validade, sequência
  e nonce) antes de aceitar;
* localmente (simulador, testes, operação de campo sem nuvem), com
  :meth:`MissionManager.start` direto.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from valtexa_rover.clock import Clock
from valtexa_rover.runtime.store import LocalStore


class MissionError(RuntimeError):
    pass


class CommandRejectedError(MissionError):
    """Comando que não passou numa das conferências. Não se executa; reporta-se."""


@dataclass(frozen=True, slots=True)
class Mission:
    id: str
    field_id: str | None
    state: str
    started_at: str
    frames_done: int
    events: int
    model_key: str | None
    model_version: str | None
    payload: dict


class MissionManager:
    def __init__(self, store: LocalStore, clock: Clock) -> None:
        self.store = store
        self.clock = clock

    def current(self) -> Mission | None:
        linhas = self.store.query(
            "SELECT * FROM missions WHERE state = 'EM_EXECUCAO' ORDER BY started_at DESC LIMIT 1"
        )
        return _mission(linhas[0]) if linhas else None

    def get(self, mission_id: str) -> Mission | None:
        linhas = self.store.query("SELECT * FROM missions WHERE id = ?", (mission_id,))
        return _mission(linhas[0]) if linhas else None

    def start(self, mission_id: str, field_id: str | None, payload: dict | None = None) -> Mission:
        if self.current() is not None:
            raise MissionError(f"Já há missão em execução ({self.current().id}).")
        if self.get(mission_id) is not None:
            raise MissionError(f"Missão {mission_id} já foi executada neste robô.")
        agora = self.clock.now().isoformat()
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO missions (id, field_id, state, started_at, payload, updated_at) "
                "VALUES (?, ?, 'EM_EXECUCAO', ?, ?, ?)",
                (mission_id, field_id, agora, json.dumps(payload or {}), agora),
            )
        return self.get(mission_id)

    def set_model(self, mission_id: str, key: str, version: str, sha256: str) -> None:
        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE missions SET model_key = ?, model_version = ?, model_sha256 = ? "
                "WHERE id = ?",
                (key, version, sha256, mission_id),
            )

    def progress(self, mission_id: str, frames: int, events: int) -> None:
        """Soma progresso. Chamado **dentro** da transação do quadro."""
        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE missions SET frames_done = frames_done + ?, events = events + ?, "
                "updated_at = ? WHERE id = ?",
                (frames, events, self.clock.now().isoformat(), mission_id),
            )

    def finish(self, mission_id: str, state: str = "CONCLUIDA") -> None:
        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE missions SET state = ?, ended_at = ?, updated_at = ? WHERE id = ?",
                (state, self.clock.now().isoformat(), self.clock.now().isoformat(), mission_id),
            )


def canonical_command(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


class PisoDaPlataforma:
    """Limite inferior **assinado** da hora da plataforma, contra o relógio do robô errado.

    Todo envelope assinado traz ``issued_at`` (hora da plataforma quando assinou). Depois de
    conferida a assinatura, ele prova que a hora da plataforma *agora* é pelo menos
    ``issued_at`` + o tempo monotônico que passou desde que o vimos. A validade
    (``expires_at``) passa a ser conferida contra ``max(relógio do robô, piso)``.

    Assim um ponto de teleop (ou comando) **atrasado** — guardado por uma base com defeito ou
    hostil, ou preso numa fila — é recusado mesmo com o relógio do robô **atrasado** (sem
    GNSS ainda, RTC errado), desde que o robô já tenha visto um envelope mais novo. O teleop
    é buscado a cada ciclo, então o piso anda junto. Relógio do robô **adiantado** só recusa
    o que valia (para o lado seguro); o ensaio ``relogio`` confere os dois contra o GNSS.

    ``monotonic``: o tempo que **passou de verdade** (padrão ``time.monotonic``). Não é o do
    relógio do Runtime: na bancada simulada o monotônico dele anda depressa (as esperas
    são virtuais) e empurraria o piso para o futuro.
    """

    def __init__(self, monotonic: Callable[[], float] = time.monotonic) -> None:
        self.monotonic = monotonic
        self._issued: datetime | None = None
        self._mono = 0.0

    def observa(self, issued_at: str | None) -> None:
        try:
            emitido = datetime.fromisoformat(str(issued_at))
        except ValueError:
            return
        if emitido.tzinfo is None:
            return
        piso = self.piso()
        if piso is None or emitido > piso:
            self._issued, self._mono = emitido, self.monotonic()

    def piso(self) -> datetime | None:
        if self._issued is None:
            return None
        return self._issued + timedelta(seconds=max(0.0, self.monotonic() - self._mono))

    def agora(self, now: datetime) -> datetime:
        piso = self.piso()
        return now if piso is None or piso <= now else piso

    def atraso_s(self, now: datetime) -> float:
        """Quanto o relógio do robô está, no mínimo, atrás da plataforma (0 se não sabe)."""
        piso = self.piso()
        return 0.0 if piso is None else max(0.0, (piso - now).total_seconds())


def verify_command(
    envelope: dict, *, device_id: str, platform_public_key: str, store: LocalStore, now: datetime,
    sequence_key: str = "last_command_sequence", command_prefix: str | None = None,
    piso: PisoDaPlataforma | None = None,
) -> dict:
    """As cinco conferências do protocolo; devolve o ``body`` ou levanta CommandRejectedError.

    ``sequence_key``: cada canal tem a sua sequência (missão, comando remoto, controle
    remoto), para um não recusar o outro. ``command_prefix``: o tipo esperado
    (``device.``), para um envelope de missão não passar por comando remoto e vice-versa.
    ``piso``: limite inferior assinado da hora da plataforma (:class:`PisoDaPlataforma`).
    """
    payload = envelope.get("payload") or {}
    try:
        chave = Ed25519PublicKey.from_public_bytes(bytes.fromhex(platform_public_key))
        chave.verify(bytes.fromhex(envelope.get("signature", "")), canonical_command(payload))
    except (InvalidSignature, ValueError) as exc:
        raise CommandRejectedError("assinatura inválida") from exc
    if piso is not None:
        piso.observa(payload.get("issued_at"))
        now = piso.agora(now)
    if payload.get("device_id") != device_id:
        raise CommandRejectedError("comando endereçado a outro robô")
    if command_prefix and not str(payload.get("command", "")).startswith(command_prefix):
        raise CommandRejectedError(f"tipo de comando inesperado: {payload.get('command')!r}")
    expira = datetime.fromisoformat(payload["expires_at"])
    if expira <= now:
        raise CommandRejectedError("comando vencido")
    ultima = int(store.get(sequence_key, 0))
    if int(payload.get("sequence", 0)) <= ultima:
        raise CommandRejectedError("sequência não é maior que a do último comando aceito")
    vistos = set(store.get("command_nonces", []))
    if payload.get("nonce") in vistos:
        raise CommandRejectedError("nonce repetido")
    with store.transaction():
        store.put(sequence_key, int(payload["sequence"]))
        store.put("command_nonces", [*list(vistos)[-200:], payload.get("nonce")])
    return payload.get("body") or {}


def _mission(linha) -> Mission:
    return Mission(
        id=linha["id"], field_id=linha["field_id"], state=linha["state"],
        started_at=linha["started_at"], frames_done=linha["frames_done"],
        events=linha["events"], model_key=linha["model_key"],
        model_version=linha["model_version"], payload=json.loads(linha["payload"]),
    )
