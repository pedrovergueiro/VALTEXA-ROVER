"""Event Engine: eventos do Rover (brief, seção 40), gravados na fila local.

Cada evento tem um ID (UUID) gerado **no robô** e o SHA-256 do JSON canônico do
conteúdo. É por esse par que a nuvem deduplica: o mesmo evento enviado duas
vezes vira uma ocorrência só (seção 52); o mesmo ID com conteúdo diferente é
recusado (``409``), porque aí não é reenvio — é corrupção ou bug.

Os tipos da seção 40 que nascem no Rover estão em :class:`EventType`. Os demais
(PRESCRIPTION_*, APPLICATION_REGISTERED, VERIFICATION_COMPLETED) nascem na
plataforma e não passam por aqui. ``ROVER_ONLINE``/``ROVER_OFFLINE`` são
registrados aqui quando o enlace muda, e a plataforma também os deriva do
contato (frota).
"""

from __future__ import annotations

import json
import uuid

from valtexa_rover._compat import StrEnum
from valtexa_rover.clock import Clock
from valtexa_rover.config import Priority
from valtexa_rover.runtime.store import LocalStore, canonical_json, sha256_hex


class EventType(StrEnum):
    MISSION_STARTED = "MISSION_STARTED"
    MISSION_RESUMED = "MISSION_RESUMED"
    MISSION_COMPLETED = "MISSION_COMPLETED"
    IMAGE_EVENT_DETECTED = "IMAGE_EVENT_DETECTED"
    AI_RESULT_CREATED = "AI_RESULT_CREATED"
    UPLOAD_STARTED = "UPLOAD_STARTED"
    UPLOAD_COMPLETED = "UPLOAD_COMPLETED"
    UPLOAD_FAILED = "UPLOAD_FAILED"
    MODEL_UPDATED = "MODEL_UPDATED"
    MODEL_REJECTED = "MODEL_REJECTED"
    ROVER_ONLINE = "ROVER_ONLINE"
    ROVER_OFFLINE = "ROVER_OFFLINE"
    HARDWARE_FAULT = "HARDWARE_FAULT"
    STORAGE_PRESSURE = "STORAGE_PRESSURE"
    # OTA (seção 24): cada passo da atualização remota
    OTA_DOWNLOADED = "OTA_DOWNLOADED"
    OTA_INSTALLED = "OTA_INSTALLED"
    OTA_APPLIED = "OTA_APPLIED"
    OTA_ROLLED_BACK = "OTA_ROLLED_BACK"
    OTA_REJECTED = "OTA_REJECTED"
    OTA_FAILED = "OTA_FAILED"


SEVERIDADE_PADRAO = {
    EventType.UPLOAD_FAILED: "warning",
    EventType.MODEL_REJECTED: "critical",
    EventType.HARDWARE_FAULT: "warning",
    EventType.STORAGE_PRESSURE: "warning",
    EventType.OTA_ROLLED_BACK: "critical",
    EventType.OTA_REJECTED: "critical",
    EventType.OTA_FAILED: "warning",
}

PRIORIDADE_PADRAO = {
    EventType.MISSION_STARTED: Priority.CONCLUSAO_MISSAO,
    EventType.MISSION_RESUMED: Priority.CONCLUSAO_MISSAO,
    EventType.MISSION_COMPLETED: Priority.CONCLUSAO_MISSAO,
    EventType.MODEL_UPDATED: Priority.CONCLUSAO_MISSAO,
    EventType.UPLOAD_STARTED: Priority.SECUNDARIO,
    EventType.UPLOAD_COMPLETED: Priority.SECUNDARIO,
    EventType.ROVER_ONLINE: Priority.SECUNDARIO,
    EventType.ROVER_OFFLINE: Priority.SECUNDARIO,
    EventType.OTA_DOWNLOADED: Priority.SECUNDARIO,
    EventType.OTA_INSTALLED: Priority.CONCLUSAO_MISSAO,
    EventType.OTA_APPLIED: Priority.CONCLUSAO_MISSAO,
}


class EventEngine:
    def __init__(self, store: LocalStore, clock: Clock, device_id: str) -> None:
        self.store = store
        self.clock = clock
        self.device_id = device_id

    def emit(
        self,
        event_type: EventType,
        payload: dict | None = None,
        *,
        mission_id: str | None = None,
        severity: str | None = None,
        priority: Priority | None = None,
        event_id: str | None = None,
    ) -> str:
        """Grava o evento e o põe na fila, na transação corrente (ou numa própria)."""
        event_id = event_id or str(uuid.uuid4())
        severidade = severity or SEVERIDADE_PADRAO.get(event_type, "info")
        if priority is None:
            priority = (
                Priority.EVENTO_CRITICO
                if severidade == "critical"
                else PRIORIDADE_PADRAO.get(event_type, Priority.ANOMALIA)
            )
        corpo = {
            "id": event_id,
            "kind": "event",
            "type": str(event_type),
            "severity": severidade,
            "device_id": self.device_id,
            "mission_id": mission_id,
            "occurred_at": self.clock.now().isoformat(),
            "payload": payload or {},
        }
        enqueue_record(self.store, self.clock, corpo, priority)
        return event_id


def enqueue_record(store: LocalStore, clock: Clock, corpo: dict, priority: Priority) -> str:
    """Grava um registro (resultado ou evento) e o item da fila que o leva à nuvem."""
    bruto = canonical_json(corpo)
    digest = sha256_hex(bruto)
    with store.transaction() as conn:
        conn.execute(
            "INSERT INTO records (id, kind, type, mission_id, occurred_at, sha256, payload) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (corpo["id"], corpo["kind"], corpo["type"], corpo.get("mission_id"),
             corpo["occurred_at"], digest, bruto.decode()),
        )
        seq = store.next_counter("queue_seq")
        conn.execute(
            "INSERT INTO sync_queue (id, kind, ref_id, sha256, size_bytes, priority, status, "
            "created_at, seq) VALUES (?, 'record', ?, ?, ?, ?, 'PENDENTE', ?, ?)",
            (f"rec-{corpo['id']}", corpo["id"], digest, len(bruto), int(priority),
             clock.now().isoformat(), seq),
        )
    return digest


def load_record(store: LocalStore, record_id: str) -> tuple[dict, str] | None:
    linhas = store.query("SELECT payload, sha256 FROM records WHERE id = ?", (record_id,))
    if not linhas:
        return None
    return json.loads(linhas[0]["payload"]), linhas[0]["sha256"]
