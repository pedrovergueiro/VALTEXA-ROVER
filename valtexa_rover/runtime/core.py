"""Core do Rover Runtime: liga HAL, Edge AI, buffer, retenção, eventos, telemetria e sync.

O pipeline da seção 14, quadro a quadro, **sem depender de rede**::

    CÂMERA → CAPTURA → PRÉ-PROCESSAMENTO → EDGE AI → CLASSIFICAÇÃO →
    DECISÃO DE RETENÇÃO → ARMAZENAMENTO LOCAL → (fila) → SINCRONIZAÇÃO

Cada quadro termina numa transação só do banco local: resultado (metadados da
seção 16), arquivos que a retenção manda guardar, evento (se houver), quadros de
contexto do buffer circular que este quadro completou, itens da fila e o
progresso da missão. Um reinício no meio de um quadro deixa o quadro inteiro ou
nada — e a missão continua do último quadro gravado (:meth:`RoverCore.boot`).

A sincronização roda no mesmo laço, limitada a ``sync_requests_per_step``
requisições por passo, para nunca segurar a captura. Num robô real ela pode
rodar numa thread (:meth:`RoverCore.sync_forever`): o banco é protegido por
trava e as transações são curtas.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

from valtexa_rover import EDGE_AI_VERSION, RUNTIME_VERSION
from valtexa_rover.clock import Clock, SystemClock
from valtexa_rover.config import Category, Priority, RuntimeConfig, Tier
from valtexa_rover.edge_ai.engine import EdgeAIEngine, Inference
from valtexa_rover.edge_ai.vendor.preprocess import InvalidImageError
from valtexa_rover.hal.interfaces import (
    Frame,
    GpsFix,
    Hardware,
    HardwareUnavailableError,
    NetworkUnavailableError,
)
from valtexa_rover.runtime.buffer import CircularBuffer, EventSequence
from valtexa_rover.runtime.client import DeviceClient, DeviceIdentity
from valtexa_rover.runtime.events import EventEngine, EventType, enqueue_record
from valtexa_rover.runtime.health import HealthMonitor, Telemetry
from valtexa_rover.runtime.logs import setup_logging
from valtexa_rover.runtime.mission import (
    MissionError,
    MissionManager,
    PisoDaPlataforma,
    verify_command,
)
from valtexa_rover.runtime.model_manager import ArtifactSource, ModelError, ModelManager
from valtexa_rover.runtime.ota import OtaManager, OtaReport
from valtexa_rover.runtime.retention import RetentionDecision, RetentionManager, reduce_image
from valtexa_rover.runtime.store import LocalStore
from valtexa_rover.runtime.sync import SyncEngine, SyncReport

EVENT_CATEGORIES = {Category.POSSIVEL_PRAGA, Category.POSSIVEL_DOENCA, Category.ANOMALIA}
SUFFIX = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}


@dataclass(slots=True)
class StepReport:
    frames: int = 0
    events: int = 0
    camera_failures: int = 0
    files_bytes: int = 0
    categories: dict[str, int] = field(default_factory=dict)
    sync: SyncReport | None = None
    ota: OtaReport | None = None


class RoverCore:
    def __init__(
        self,
        hardware: Hardware,
        data_dir: Path | str,
        identity: DeviceIdentity,
        *,
        config: RuntimeConfig | None = None,
        clock: Clock | None = None,
        artifact_source: ArtifactSource,
        telemetry_interval_s: float = 10.0,
        purge_interval_s: float = 300.0,
        auto_sync: bool = True,
        sync_requests_per_step: int = 20,
        region_resolver: Callable[[GpsFix | None], dict | None] | None = None,
        seed: int | None = None,
        config_path: Path | str | None = None,
        ota_interval_s: float = 600.0,
    ) -> None:
        self.hardware = hardware
        self.identity = identity
        self.config = (config or RuntimeConfig()).validate()
        self.clock = clock or SystemClock()
        # hora da plataforma vista nos envelopes assinados (validade com o relógio errado)
        self.piso_plataforma = PisoDaPlataforma()
        self.log = setup_logging(data_dir)
        self.store = LocalStore(data_dir)
        rede = hardware.network
        if hasattr(rede, "chave_plataforma") and not rede.chave_plataforma:
            # rede via base: a resposta que muda estado só vale com o recibo da plataforma
            rede.chave_plataforma = identity.platform_public_key
        self.client = DeviceClient(identity, hardware.network, self.clock)
        self.events = EventEngine(self.store, self.clock, identity.device_id)
        self.engine = EdgeAIEngine(hardware.accelerator, self.config.classificacao)
        self.models = ModelManager(self.store, self.engine, self.events, artifact_source,
                                   self.client)
        self.missions = MissionManager(self.store, self.clock)
        self.retention = RetentionManager(self.config.retencao)
        self.health = HealthMonitor(hardware, self.store, self.engine, self.config.saude,
                                    self.clock)
        self.health.missao_em_curso = lambda: self.missions.current() is not None
        self.telemetry = Telemetry(self.store, self.health, self.clock)
        self.sync = SyncEngine(self.store, self.client, self.events, self.clock,
                               self.config.sync, versions=self._versions, seed=seed)
        self.health.sync_status = self.sync.saude
        self.ota = OtaManager(
            self.store, self.client, self.events, self.clock,
            public_key=identity.ota_public_key, device_id=identity.device_id,
            accelerator=hardware.accelerator, classification=self.config.classificacao,
            models_folder=getattr(artifact_source, "folder", None),
            config_path=Path(config_path) if config_path else None,
            mission_running=lambda: self.missions.current() is not None,
        )
        self.ota_interval_s = ota_interval_s
        self._last_ota: float | None = None
        self.buffers: dict[str, CircularBuffer] = {}
        self.ultimos_quadros: dict[str, Frame] = {}
        """O último quadro de cada câmera: o vídeo ao vivo usa este, sem chamar a câmera."""
        self.telemetry_interval_s = telemetry_interval_s
        self.purge_interval_s = purge_interval_s
        self.auto_sync = auto_sync
        self.sync_requests_per_step = sync_requests_per_step
        self.region_resolver = region_resolver
        self._last_telemetry: float | None = None
        self._last_purge = self.clock.monotonic()
        self.recovered: dict[str, int] = {}

    # -- ciclo de vida ----------------------------------------------------------

    def boot(self) -> None:
        """Recupera o disco e, se havia missão em execução, retoma (``MISSION_RESUMED``)."""
        self.recovered = self.store.recover()
        self.models.refresh()
        missao = self.missions.current()
        if missao is not None:
            if missao.field_id:
                self.models.activate_for_field(missao.field_id, missao.id)
            self.events.emit(
                EventType.MISSION_RESUMED,
                {"frames_done": missao.frames_done, "events": missao.events,
                 "recovered": self.recovered},
                mission_id=missao.id,
            )
            self.log.info("missão retomada", extra={"data": {"mission_id": missao.id}})
        else:
            self.precarrega_modelo()
        # Chegou até aqui: o que estava em teste (runtime, Edge AI, configuração
        # instalados por OTA) subiu inteiro e vira o atual de vez.
        confirmados = self.ota.confirm_boot()
        if confirmados:
            self.log.info("OTA confirmada no boot", extra={"data": {"kinds": confirmados}})

    def close(self) -> None:
        self.store.close()

    def precarrega_modelo(self) -> str | None:
        """Fora de missão, deixa carregado o modelo de um talhão designado.

        Na Jetson o primeiro carregamento com TensorRT compila o motor (minutos): feito no
        boot, a missão começa sem espera. Tenta primeiro o talhão da última missão (o mais
        provável de vir de novo), depois os outros. Devolve ``None`` se carregou ou se não
        há designação; senão, o motivo — que vira a falha ``model_missing``.
        """
        if self.missions.current() is not None:
            return None
        designados = [d for d in self.models.designations() if d.model_key and d.version]
        ultimo = self.store.get("last_field_id")
        designados.sort(key=lambda d: d.field_id != ultimo)
        motivo = None
        for d in designados:
            try:
                self.models.activate_for_field(d.field_id)
                motivo = None
                break
            except ModelError as exc:
                motivo = str(exc)
        self.health.modelo_indisponivel = motivo
        if motivo:
            self.log.warning("pré-carga do modelo falhou", extra={"data": {"motivo": motivo}})
        return motivo

    def start_mission(self, mission_id: str, field_id: str, payload: dict | None = None) -> None:
        """Começa uma missão. Sem modelo do registry para o talhão, **não começa**."""
        self.models.refresh()
        pacote = self.models.activate_for_field(field_id)
        self.health.modelo_indisponivel = None
        self.store.put("last_field_id", field_id)
        missao = self.missions.start(mission_id, field_id, payload)
        self.missions.set_model(missao.id, pacote.model_key, pacote.version, pacote.sha256)
        self.events.emit(
            EventType.MISSION_STARTED,
            {"field_id": field_id, "model_key": pacote.model_key,
             "model_version": pacote.version, "model_sha256": pacote.sha256,
             "dnaty_version": pacote.dnaty_version, "hardware": self.hardware.hardware_version,
             "simulated_hardware": self.hardware.simulated},
            mission_id=mission_id,
        )
        self.log.info("missão iniciada", extra={"data": {"mission_id": mission_id}})

    def accept_cloud_mission(self) -> str | None:
        """Busca a missão assinada na plataforma, confere e começa. ``None`` se não há."""
        if not self.identity.platform_public_key:
            raise MissionError("Sem a chave pública da plataforma não há como conferir missão.")
        resposta = self.client.request("GET", "/v1/device/missions/current")
        if resposta.status == 204:
            return None
        envelope = resposta.json()
        corpo = verify_command(
            envelope, device_id=self.identity.device_id,
            platform_public_key=self.identity.platform_public_key, store=self.store,
            now=self.clock.now(), piso=self.piso_plataforma,
        )
        self.start_mission(corpo["mission_id"], corpo["field_id"], {"command": corpo})
        # Confirma o recebimento à plataforma (protocolo: a missão passa a "em execução").
        self._report_mission("ack", corpo["mission_id"])
        return corpo["mission_id"]

    def complete_mission(self) -> None:
        missao = self.missions.current()
        if missao is None:
            raise MissionError("Nenhuma missão em execução.")
        with self.store.transaction():
            for buffer in self.buffers.values():
                for seq in buffer.flush():
                    self._persist_sequence(seq)
            self.missions.finish(missao.id)
            atual = self.missions.get(missao.id)
            self.events.emit(
                EventType.MISSION_COMPLETED,
                {"frames_done": atual.frames_done, "events": atual.events,
                 "pending_sync": self.sync.pending()},
                mission_id=missao.id,
            )
        self.buffers.clear()
        self.log.info("missão concluída", extra={"data": {"mission_id": missao.id}})
        if (missao.payload or {}).get("command"):
            self._report_mission("complete", missao.id)

    # -- informes de missão à plataforma ------------------------------------------

    def _report_mission(self, acao: str, mission_id: str) -> None:
        """Guarda o informe (ack/complete) e tenta mandar; offline, sai no próximo passo."""
        fila = self.store.get("mission_reports", []) or []
        fila.append({"action": acao, "mission_id": mission_id})
        self.store.put("mission_reports", fila)
        self.flush_mission_reports()

    def flush_mission_reports(self) -> int:
        fila = self.store.get("mission_reports", []) or []
        enviados = 0
        while fila:
            item = fila[0]
            corpo = ({"mission_id": item["mission_id"], "accepted": True}
                     if item["action"] == "ack" else {})
            try:
                r = self.client.request(
                    "POST", f"/v1/device/missions/{item['mission_id']}/{item['action']}",
                    json_body=corpo)
            except NetworkUnavailableError:
                break
            if r.status >= 500 or r.status in (401, 429):
                break
            # 2xx, ou 4xx definitivo (já concluída, missão abortada na plataforma)
            fila.pop(0)
            self.store.put("mission_reports", fila)
            enviados += 1
        return enviados

    # -- laço ---------------------------------------------------------------------

    def step(self) -> StepReport:
        """Um ciclo: captura de todas as câmeras, Edge AI, retenção, gravação, telemetria, sync."""
        rel = StepReport()
        missao = self.missions.current()
        gps = self._read_gps()
        if missao is not None:
            for camera in self.hardware.cameras:
                try:
                    frame = camera.capture()
                except HardwareUnavailableError as exc:
                    rel.camera_failures += 1
                    self._fault(f"camera:{camera.camera_id}", str(exc), missao.id)
                    continue
                self.health.clear(f"camera:{camera.camera_id}")
                self.ultimos_quadros[camera.camera_id] = frame
                self._process(frame, gps, missao, rel)

        agora = self.clock.monotonic()
        vence = self._last_telemetry is None or (
            agora - self._last_telemetry >= self.telemetry_interval_s
        )
        if vence:
            self.telemetry.record(gps, missao.id if missao else None,
                                  self.hardware.network.is_online())
            self._last_telemetry = agora
        if agora - self._last_purge >= self.purge_interval_s:
            self.purge()
        if self.auto_sync:
            rel.sync = self.sync.step(self.sync_requests_per_step)
            if not (rel.sync.offline or rel.sync.blocked):
                self.flush_mission_reports()
        if missao is None and self.identity.ota_public_key and (
            self._last_ota is None or agora - self._last_ota >= self.ota_interval_s
        ):
            # Atualização só fora de missão (o OTA Manager confere de novo).
            self._last_ota = agora
            rel.ota = self.ota.step()
        return rel

    def purge(self) -> dict[str, int]:
        self._last_purge = self.clock.monotonic()
        invalidar = getattr(self.hardware.storage, "invalidate", None)
        if invalidar:
            invalidar()  # a limpeza decide pelo disco de agora, não pela medida em cache
        _total, livre = self.hardware.storage.usage()
        saida = self.retention.purge(self.store, self.clock.now(), free_bytes=livre,
                                     min_free_bytes=self.config.saude.min_free_bytes)
        if invalidar:
            invalidar()
        if saida["pressao_enviados"] or saida["pressao_normais"]:
            self.events.emit(EventType.STORAGE_PRESSURE, saida)
        return saida

    def sync_forever(self, stop: threading.Event, interval_s: float = 1.0) -> None:
        """Sincronização numa thread própria (robô real). O laço de captura não espera."""
        while not stop.is_set():
            self.sync.step(self.sync_requests_per_step)
            stop.wait(interval_s)

    # -- quadro -----------------------------------------------------------------

    def _process(self, frame: Frame, gps: GpsFix | None, missao, rel: StepReport) -> None:
        try:
            inferencia = self.engine.infer(frame.data, frame.distance_mm,
                                           frame.camera_height_mm, frame.hfov_deg)
        except InvalidImageError as exc:
            self._fault(f"camera:{frame.camera_id}", f"imagem ilegível: {exc}", missao.id)
            return
        decisao = self.retention.decide(inferencia.category, inferencia.low_confidence,
                                        frame.captured_at)
        result_id = str(uuid.uuid4())
        evento = inferencia.category in EVENT_CATEGORIES
        event_id = str(uuid.uuid4()) if evento else None

        # Os arquivos vão ao disco antes da transação (atômicos, com fsync). Se o
        # robô cair antes do commit, sobram sem linha e o boot os apaga.
        arquivos = self._files_for(frame, decisao)
        sequencias_prontas: list[EventSequence] = []
        with self.store.transaction() as conn:
            fila_arquivos = []
            for papel, caminho, digest, tamanho, tipo in arquivos:
                file_id = str(uuid.uuid4())
                conn.execute(
                    "INSERT INTO files (id, record_id, role, path, sha256, size_bytes, "
                    "content_type, tier, retention_until, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (file_id, result_id, papel, caminho, digest, tamanho, tipo,
                     decisao.tier.value, decisao.retention_until.isoformat(),
                     self.clock.now().isoformat()),
                )
                fila_arquivos.append((file_id, digest, tamanho, decisao.priority))
                rel.files_bytes += tamanho
            corpo = self._result_body(result_id, frame, gps, missao, inferencia, decisao,
                                      event_id, arquivos)
            enqueue_record(self.store, self.clock, corpo, decisao.priority)
            for file_id, digest, tamanho, prioridade in fila_arquivos:
                self._enqueue_file(conn, file_id, digest, tamanho, prioridade)
            if evento:
                self.events.emit(
                    EventType.IMAGE_EVENT_DETECTED,
                    {"result_id": result_id, "category": inferencia.category.value,
                     "label": inferencia.label, "confidence": inferencia.confidence,
                     "low_confidence": inferencia.low_confidence,
                     "camera_id": frame.camera_id, "field_id": missao.field_id,
                     "lat": gps.lat if gps else None, "lon": gps.lon if gps else None,
                     "image_sha256": arquivos[0][2] if arquivos else None,
                     "model_key": self.engine.package.model_key,
                     "model_version": self.engine.package.version},
                    mission_id=missao.id, severity="warning", priority=decisao.priority,
                    event_id=event_id,
                )
            buffer = self.buffers.setdefault(frame.camera_id, CircularBuffer(self.config.buffer))
            gatilho = (event_id, result_id) if evento and decisao.preserve_sequence else None
            sequencias_prontas = buffer.observe(frame, result_id, gatilho)
            for seq in sequencias_prontas:
                rel.files_bytes += self._persist_sequence(seq)
            self.missions.progress(missao.id, 1, 1 if evento else 0)
        rel.frames += 1
        rel.events += 1 if evento else 0
        chave = inferencia.category.value + ("+baixa" if inferencia.low_confidence else "")
        rel.categories[chave] = rel.categories.get(chave, 0) + 1

    NAVIGATION_KEYS = ("row_left", "row_right", "x_m", "segment_index")
    """O que a navegação diz da posição no talhão e vai no resultado (a fusão por planta usa
    a linha de cada lado e o metro ao longo dela; sem isso, a nuvem estima pelo GNSS)."""

    def _navigation(self) -> dict | None:
        if self.hardware.controller is None:
            return None
        try:
            estado = self.hardware.controller.state()
        except Exception:  # a navegação é de outro processo: falhar aqui não perde o quadro
            return None
        saida = {k: estado[k] for k in self.NAVIGATION_KEYS if estado.get(k) is not None}
        return saida or None

    def _tiles_body(self, inf: Inference, frame: Frame | None = None) -> dict | None:
        """Os recortes do quadro, para a nuvem fundir por planta (ver ``edge_ai.perfil``).

        A imagem normal sobe reduzida (320 px por padrão): pequena demais para a nuvem
        rodar o modelo de novo. Por isso as probabilidades de cada recorte vão aqui.
        """
        perfil = self.engine.profile
        if not inf.tiles or perfil is None or perfil.tile_probabilities == "none":
            return None
        rotulos = list(self.engine.labels)
        corpo = {"tile_mm": inf.tile_mm, "distance_mm": inf.distance_mm,
                 "distance_measured": inf.distance_measured,
                 "hfov_deg": inf.hfov_deg or perfil.hfov_deg,
                 "camera_height_mm": frame.camera_height_mm if frame else None,
                 "side": frame.side if frame else None, "labels": rotulos,
                 "boxes": [list(t.box) for t in inf.tiles],
                 "heights_mm": [t.height_mm for t in inf.tiles],
                 "zones": [list(t.zones) for t in inf.tiles]}
        if perfil.tile_probabilities == "full":
            corpo["probabilities"] = [list(t.probabilities) for t in inf.tiles]
        else:
            corpo["top"] = [
                [rotulos[t.probabilities.index(max(t.probabilities))], max(t.probabilities)]
                for t in inf.tiles
            ]
        return corpo

    def _files_for(self, frame: Frame, decisao: RetentionDecision) -> list[tuple]:
        if decisao.tier is Tier.METADADOS:
            return []
        if decisao.tier is Tier.ORIGINAL:
            caminho, digest = self.store.write_blob(frame.data, SUFFIX.get(frame.content_type,
                                                                          ".bin"))
            return [("original", caminho, digest, len(frame.data), frame.content_type)]
        dados, _w, _h = reduce_image(frame.data, decisao.rule.max_side, decisao.rule.jpeg_quality)
        caminho, digest = self.store.write_blob(dados, ".jpg")
        return [("reduzida", caminho, digest, len(dados), "image/jpeg")]

    def _persist_sequence(self, seq: EventSequence) -> int:
        """Grava os quadros de contexto de uma sequência do buffer (resolução do buffer)."""
        cfg = self.config.buffer
        total = 0
        with self.store.transaction() as conn:
            for posicao, quadro in seq.context:
                dados, _w, _h = reduce_image(quadro.frame.data, cfg.max_side, cfg.jpeg_quality)
                caminho, digest = self.store.write_blob(dados, ".jpg")
                file_id = str(uuid.uuid4())
                conn.execute(
                    "INSERT INTO files (id, record_id, role, path, sha256, size_bytes, "
                    "content_type, tier, retention_until, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, 'image/jpeg', ?, ?, ?)",
                    (file_id, seq.event_id, f"contexto{posicao}", caminho, digest, len(dados),
                     Tier.ORIGINAL.value, self._context_retention(quadro.frame),
                     self.clock.now().isoformat()),
                )
                self._enqueue_file(conn, file_id, digest, len(dados), cfg.priority)
                total += len(dados)
        return total

    def _context_retention(self, frame: Frame) -> str:
        """O contexto de um evento fica o mesmo tempo que o original do evento."""
        dias = [r.retention_days for r in self.config.retencao.rules.values()
                if r.preserve_sequence]
        return (frame.captured_at + timedelta(days=max(dias, default=30))).isoformat()

    def _enqueue_file(self, conn, file_id: str, digest: str, tamanho: int,
                      prioridade: Priority) -> None:
        tamanho_chunk = self.config.sync.chunk_size
        conn.execute(
            "INSERT INTO sync_queue (id, kind, ref_id, sha256, size_bytes, chunk_size, "
            "chunks_total, priority, status, created_at, seq) "
            "VALUES (?, 'file', ?, ?, ?, ?, ?, ?, 'PENDENTE', ?, ?)",
            (f"up-{file_id}", file_id, digest, tamanho, tamanho_chunk,
             max(1, -(-tamanho // tamanho_chunk)), int(prioridade), self.clock.now().isoformat(),
             self.store.next_counter("queue_seq")),
        )

    def _result_body(self, result_id, frame, gps, missao, inf: Inference,
                     decisao: RetentionDecision, event_id, arquivos) -> dict:
        pacote = self.engine.package
        sensores = {}
        if self.hardware.sensors is not None:
            try:
                sensores = dict(self.hardware.sensors.read().values)
            except HardwareUnavailableError:
                sensores = {}
        imu = None
        if self.hardware.imu is not None:
            try:
                leitura = self.hardware.imu.read()
                imu = {"roll_deg": leitura.roll_deg, "pitch_deg": leitura.pitch_deg,
                       "yaw_deg": leitura.yaw_deg}
            except HardwareUnavailableError:
                imu = None
        return {
            "id": result_id,
            "navigation": self._navigation(),
            "kind": "result",
            "type": inf.category.value,
            "device_id": self.identity.device_id,
            "mission_id": missao.id,
            "field_id": missao.field_id,
            "occurred_at": frame.captured_at.isoformat(),
            "frame": {"camera_id": frame.camera_id, "frame_index": frame.frame_index,
                      "width": frame.width, "height": frame.height,
                      "content_type": frame.content_type},
            "position": None if gps is None else {
                "lat": gps.lat, "lon": gps.lon, "alt_m": gps.alt_m, "fix": gps.fix,
                "hdop": gps.hdop, "heading_deg": gps.heading_deg,
            },
            "region": self.region_resolver(gps) if self.region_resolver else None,
            "imu": imu,
            "sensors": sensores,
            "model": {"key": pacote.model_key, "version": pacote.version,
                      "sha256": pacote.sha256, "artifact_kind": pacote.artifact_kind,
                      "dnaty_version": pacote.dnaty_version,
                      "target_hardware": pacote.target_hardware},
            "hardware": {"version": self.hardware.hardware_version,
                         "accelerator": self.hardware.accelerator.describe().model,
                         "providers": self.hardware.accelerator.providers(),
                         "simulated": self.hardware.simulated},
            "result": {"category": inf.category.value, "label": inf.label,
                       "confidence": inf.confidence, "low_confidence": inf.low_confidence,
                       "top": [[rotulo, p] for rotulo, p in inf.top], "reason": inf.reason,
                       "latency_ms": inf.latency_ms, "input_size": inf.input_size,
                       "event_tiles": inf.event_tiles, "evidence": inf.evidence},
            "tiles": self._tiles_body(inf, frame),
            "retention": {"tier": decisao.tier.value,
                          "retention_until": decisao.retention_until.isoformat(),
                          "priority": int(decisao.priority),
                          "sampled_for_validation": decisao.sampled_for_validation},
            "event_id": event_id,
            "files": [{"role": papel, "sha256": digest, "size_bytes": tamanho}
                      for papel, _c, digest, tamanho, _t in arquivos],
            "runtime_version": RUNTIME_VERSION,
            "edge_ai_version": EDGE_AI_VERSION,
        }

    # -- apoio --------------------------------------------------------------------

    def _read_gps(self) -> GpsFix | None:
        try:
            fix = self.hardware.gps.read()
        except HardwareUnavailableError as exc:
            self._fault("gps", str(exc), None)
            return None
        self.health.clear("gps")
        return fix

    def _fault(self, code: str, detail: str, mission_id: str | None) -> None:
        if self.health.fault(code, detail):
            self.events.emit(EventType.HARDWARE_FAULT, {"code": code, "detail": detail},
                             mission_id=mission_id)
            self.log.warning("falha de hardware", extra={"data": {"code": code,
                                                                  "detail": detail}})

    def _versions(self) -> dict:
        pacote = self.engine.package
        return {
            "model_key": pacote.model_key if pacote else None,
            "model_version": pacote.version if pacote else None,
            "dnaty_version": pacote.dnaty_version if pacote else None,
            "hardware_version": self.hardware.hardware_version,
        }

    def summary(self) -> dict:
        return {
            "records": self.store.scalar("SELECT COUNT(*) FROM records"),
            "files": self.store.scalar("SELECT COUNT(*) FROM files"),
            "files_bytes": self.store.scalar("SELECT COALESCE(SUM(size_bytes), 0) FROM files"),
            "queue": self.sync.queue_status(),
            "pending": self.sync.pending(),
        }


__all__ = ["ModelError", "RoverCore", "StepReport"]
