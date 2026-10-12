"""Model Manager: o Rover só roda modelo do registry, com hash conferido (seções 9 e 22).

O ciclo:

1. :meth:`ModelManager.refresh` pergunta à plataforma ``GET /v1/device/models``
   — para cada talhão, a versão em PRODUÇÃO do Model Registry, com o SHA-256 do
   artefato (o otimizado pelo dNATY quando houver). A resposta fica gravada no
   banco local: offline, o robô usa a última designação que recebeu;
2. o artefato chega por um :class:`ArtifactSource` (hoje: uma pasta local, que
   em campo é a área de *staging* do OTA da Parte 5 ou um pendrive de serviço);
3. :meth:`ModelManager.activate_for_field` confere o SHA-256 **do arquivo** contra
   o do registry antes de entregar à Edge AI. Hash diferente → ``MODEL_REJECTED``
   (crítico) e o modelo não carrega. Designação sem hash → não carrega.
   Arquivo que não está designado → nunca carrega, nem que esteja na pasta.

O que o robô roda vai para a plataforma em ``POST /v1/device/session`` (a frota
mostra "conforme/divergente/fora do registry") e em **cada resultado** da Edge AI.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from valtexa_rover.edge_ai.engine import EdgeAIEngine, EdgeAIError, ModelPackage
from valtexa_rover.hal.interfaces import NetworkUnavailableError
from valtexa_rover.runtime.client import DeviceClient
from valtexa_rover.runtime.events import EventEngine, EventType
from valtexa_rover.runtime.store import LocalStore


class ModelError(RuntimeError):
    """Sem modelo utilizável do registry para este talhão. A mensagem diz por quê."""


@dataclass(frozen=True, slots=True)
class Designation:
    field_id: str
    model_key: str | None
    version: str | None
    artifact_uri: str | None
    artifact_sha256: str | None
    artifact_kind: str | None
    dnaty_version: str | None
    target_hardware: str | None
    reason: str | None = None


class ArtifactSource(Protocol):
    """De onde vem o arquivo do modelo. Não confere nada: quem confere é o Manager."""

    def fetch(self, designation: Designation, destination: Path) -> bool:
        """Copia o artefato para ``destination`` (e os rótulos ao lado). ``False`` se não tem."""
        ...


class FolderArtifactSource:
    """Artefatos numa pasta local, nomeados pelo SHA-256 (``<sha>.onnx`` + ``.labels.json``).

    Em campo é a área de *staging* do OTA (Parte 5) ou um pendrive de serviço.
    Nada aqui é confiável por estar na pasta: o hash é conferido depois.
    """

    def __init__(self, folder: Path | str) -> None:
        self.folder = Path(folder)

    def fetch(self, designation: Designation, destination: Path) -> bool:
        if not designation.artifact_sha256:
            return False
        origem = self.folder / f"{designation.artifact_sha256}.onnx"
        if not origem.exists():
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origem, destination)
        for extra in EXTRAS:
            lado = origem.with_suffix(extra)
            if lado.exists():
                shutil.copyfile(lado, destination.with_suffix(extra))
        return True


EXTRAS = (".labels.json", ".categories.json", ".edge.json", ".head.npz", ".encoder.onnx")
"""Arquivos que viajam ao lado do ``<sha>.onnx``. ``.head.npz`` e ``.encoder.onnx`` (a torre
de visão do SigLIP 2, ~355 MB) só existem nos modelos com cabeça — o café."""


def stage_model(onnx: Path | str, folder: Path | str, engine: EdgeAIEngine, *,
                model_key: str | None = None, require_profile: bool = True,
                encoder: Path | str | None = None) -> dict:
    """Confere um modelo e o põe na área de *staging* com o nome que o OTA procura.

    Monta ``<sha256>.onnx`` e os arquivos ao lado numa pasta de preparo dentro do destino,
    carrega **de lá** (rótulos, categorias, perfil ``.edge.json``, cabeça e torre de visão
    quando o perfil pede, grafo no acelerador deste computador) e só então move para o
    lugar: um modelo que não carrega aqui não carregaria em missão, e nada pela metade
    fica no staging. ``encoder`` é a torre de visão (``framing_vision.onnx`` do ``ia/``)
    quando ela não está ao lado do modelo como ``<modelo>.encoder.onnx``. O registry da
    plataforma designa o modelo para o talhão pelo SHA-256 impresso.
    """
    origem = Path(onnx)
    chave = model_key or origem.stem
    sha = file_sha256(origem)
    if require_profile and not origem.with_suffix(".edge.json").exists():
        raise ModelError(
            f"{origem.with_suffix('.edge.json').name} não existe: sem o perfil calibrado "
            "(training.fit_edge) o modelo classificaria o quadro inteiro, não recortes de 150 mm."
        )
    destino = Path(folder)
    preparo = destino / f".preparo-{sha[:12]}"
    if preparo.exists():
        shutil.rmtree(preparo)
    preparo.mkdir(parents=True)
    try:
        fontes = [(origem, ".onnx")] + [(origem.with_suffix(x), x) for x in EXTRAS
                                        if origem.with_suffix(x).exists()]
        if encoder is not None and not origem.with_suffix(".encoder.onnx").exists():
            fontes.append((Path(encoder), ".encoder.onnx"))
        for de, sufixo in fontes:
            shutil.copyfile(de, preparo / f"{sha}{sufixo}")
        try:
            engine.load(ModelPackage(chave, "preparo", sha, preparo / f"{sha}.onnx"))
        except EdgeAIError as exc:
            raise ModelError(f"{origem.name} não carrega: {exc}") from exc
        copiados = []
        for arquivo in sorted(preparo.iterdir()):
            arquivo.replace(destino / arquivo.name)
            copiados.append(arquivo.name)
    finally:
        shutil.rmtree(preparo, ignore_errors=True)
    return {"model_key": chave, "sha256": sha, "labels": list(engine.labels),
            "profile": engine.profile is not None, "head": engine.head is not None,
            "files": copiados, "folder": str(destino)}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for bloco in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(bloco)
    return h.hexdigest()


class ModelManager:
    def __init__(
        self,
        store: LocalStore,
        engine: EdgeAIEngine,
        events: EventEngine,
        source: ArtifactSource,
        client: DeviceClient | None = None,
    ) -> None:
        self.store = store
        self.engine = engine
        self.events = events
        self.source = source
        self.client = client
        self.models_dir = store.root / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

    # -- designação -----------------------------------------------------------

    def refresh(self) -> bool:
        """Atualiza a designação pela plataforma. ``False`` se offline (usa a guardada)."""
        if self.client is None:
            return False
        try:
            resposta = self.client.request("GET", "/v1/device/models")
        except NetworkUnavailableError:
            return False
        if resposta.status != 200:
            return False
        self.store.put("designations", resposta.json())
        return True

    def designations(self) -> list[Designation]:
        bruto = self.store.get("designations", []) or []
        return [
            Designation(
                field_id=item["field_id"],
                model_key=item.get("model_key"),
                version=item.get("version"),
                artifact_uri=item.get("artifact_uri"),
                artifact_sha256=item.get("artifact_sha256"),
                artifact_kind=item.get("artifact_kind"),
                dnaty_version=item.get("dnaty_version"),
                target_hardware=item.get("target_hardware"),
                reason=item.get("reason"),
            )
            for item in bruto
        ]

    def designation_for(self, field_id: str) -> Designation:
        for d in self.designations():
            if d.field_id == field_id:
                return d
        raise ModelError(
            f"O registry não designou modelo para o talhão {field_id} (ou o robô nunca "
            "recebeu a designação: conecte uma vez antes da primeira missão)."
        )

    # -- ativação -------------------------------------------------------------

    _marca_carregada: str | None = None

    @staticmethod
    def _marca_dos_extras(destino: Path) -> str | None:
        """Hash do perfil e da cabeça que viajam ao lado do ONNX (mudam sem mudar o ONNX)."""
        partes = [file_sha256(destino.with_suffix(x)) for x in (".edge.json", ".head.npz")
                  if destino.with_suffix(x).exists()]
        return ":".join(partes) or None

    def activate_for_field(self, field_id: str, mission_id: str | None = None) -> ModelPackage:
        d = self.designation_for(field_id)
        if not d.model_key or not d.version:
            raise ModelError(f"Talhão {field_id} sem modelo no registry: {d.reason or '—'}.")
        if not d.artifact_sha256:
            raise ModelError(
                f"{d.model_key} {d.version} está no registry sem hash de artefato: o robô "
                "não carrega modelo que não pode conferir."
            )
        destino = self.models_dir / f"{d.artifact_sha256}.onnx"
        atual = self.engine.package
        # O mesmo ONNX com perfil recalibrado (limiares da calibração de campo) é outro
        # modelo na prática: recarrega. Sem isto o perfil novo só valeria no próximo boot.
        marca = self._marca_dos_extras(destino)
        if (atual is not None and atual.sha256 == d.artifact_sha256
                and marca == self._marca_carregada):
            return atual

        if not destino.exists() and not self.source.fetch(d, destino):
            raise ModelError(
                f"Artefato de {d.model_key} {d.version} ({d.artifact_sha256[:12]}…) não está "
                "no robô nem na origem de artefatos."
            )
        calculado = file_sha256(destino)
        if calculado != d.artifact_sha256:
            destino.unlink(missing_ok=True)
            self.events.emit(
                EventType.MODEL_REJECTED,
                {"model_key": d.model_key, "version": d.version,
                 "expected_sha256": d.artifact_sha256, "actual_sha256": calculado},
                mission_id=mission_id,
            )
            raise ModelError(
                f"Hash do artefato de {d.model_key} {d.version} não confere com o registry: "
                "modelo recusado."
            )
        pacote = ModelPackage(
            model_key=d.model_key,
            version=d.version,
            sha256=calculado,
            path=destino,
            artifact_kind=d.artifact_kind or "original",
            dnaty_version=d.dnaty_version,
            target_hardware=d.target_hardware,
        )
        anterior = self.engine.package
        registrado = self.store.get("active_model") or {}
        try:
            self.engine.load(pacote)
            self._marca_carregada = marca
        except EdgeAIError as exc:
            self.events.emit(
                EventType.MODEL_REJECTED,
                {"model_key": d.model_key, "version": d.version, "reason": str(exc)},
                mission_id=mission_id,
            )
            if anterior is not None:
                self.engine.load(anterior)
            raise ModelError(str(exc)) from exc
        if registrado.get("sha256") == pacote.sha256:
            return pacote  # mesmo modelo de antes do reinício: não é atualização
        self.store.put("active_model", {
            "model_key": pacote.model_key, "version": pacote.version, "sha256": pacote.sha256,
            "artifact_kind": pacote.artifact_kind, "dnaty_version": pacote.dnaty_version,
        })
        self.events.emit(
            EventType.MODEL_UPDATED,
            {"model_key": pacote.model_key, "version": pacote.version, "sha256": pacote.sha256,
             "previous": registrado.get("version") and
             f"{registrado.get('model_key')}@{registrado.get('version')}",
             "artifact_kind": pacote.artifact_kind, "dnaty_version": pacote.dnaty_version},
            mission_id=mission_id,
        )
        return pacote
