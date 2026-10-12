"""OTA Manager: atualização remota assinada, instalada lado a lado, com *rollback* (seção 24).

O ciclo de um pacote (``runtime``, ``edge_ai``, ``model`` ou ``config``)::

    GET /v1/device/ota → envelope assinado (Ed25519, chave OTA da plataforma)
      → confere assinatura, destinatário, validade, tipo e se já foi processado
      → baixa em pedaços para ``ota/staging/<id>.tar.part`` (retoma do byte em que parou)
      → confere tamanho e SHA-256 do pacote e de cada arquivo contra o manifesto
      → instala **lado a lado** em ``ota/slots/<tipo>/<versão>/`` (o atual não é tocado)
      → confere a saúde do novo (modelo: carrega e infere; configuração: valida;
        runtime/Edge AI: compila tudo e importa o pacote, versão igual à do manifesto)
      → troca o ponteiro ``ota/current.json`` de forma atômica, guardando o anterior
      → runtime/Edge AI/configuração ficam "em teste" até o próximo boot confirmar
        (:meth:`OtaManager.on_boot` / :meth:`OtaManager.confirm_boot`); se o novo não
        sobe em ``max_trial_boots`` boots, volta sozinho ao anterior

Qualquer falha antes da troca deixa o atual como está (``falhou``/``recusada``);
falha depois da troca volta ao anterior (``revertida``). Cada passo vira evento
local (``OTA_*``, sobe pela fila) e é informado à plataforma
(``POST /v1/device/ota/{id}/status``); offline, o informe fica guardado e sai
quando a rede voltar. **Nada é instalado com missão em execução.**

A chave pública OTA chega no provisionamento (``ota_public_key``); a privada
mora só na plataforma, fora do código (``OTA_SIGNING_SECRET``).
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from valtexa_rover.clock import Clock
from valtexa_rover.config import ClassificationConfig, ConfigError, RuntimeConfig
from valtexa_rover.edge_ai.engine import EdgeAIEngine, ModelPackage
from valtexa_rover.hal.interfaces import AiAccelerator, NetworkUnavailableError
from valtexa_rover.runtime.client import DeviceClient
from valtexa_rover.runtime.events import EventEngine, EventType
from valtexa_rover.runtime.store import LocalStore

OTA_ALGORITHM = "VALTEXA-ED25519-OTA-V1"
KINDS = ("runtime", "edge_ai", "model", "config")
COM_TESTE_DE_BOOT = ("runtime", "edge_ai", "config")


class OtaRejectedError(RuntimeError):
    """O pacote não é confiável (assinatura, destinatário, validade, hash). Não se instala."""


class OtaInstallError(RuntimeError):
    """O pacote é legítimo, mas não instalou ou não passou na saúde."""


def canonical_ota(payload: dict) -> bytes:
    corpo = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return (OTA_ALGORITHM + "\n" + corpo).encode("utf-8")


def verify_envelope(envelope: dict, *, public_key: str, device_id: str, now: datetime) -> dict:
    """Confere o envelope e devolve o ``payload``. Levanta :class:`OtaRejectedError`."""
    try:
        payload = envelope["payload"]
        assinatura = bytes.fromhex(envelope["signature"])
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key)).verify(
            assinatura, canonical_ota(payload)
        )
    except (KeyError, ValueError, TypeError, InvalidSignature) as exc:
        raise OtaRejectedError("assinatura do pacote inválida") from exc
    if payload.get("device_id") != device_id:
        raise OtaRejectedError("pacote destinado a outro robô")
    try:
        expira = datetime.fromisoformat(payload["expires_at"])
    except (KeyError, ValueError) as exc:
        raise OtaRejectedError("envelope sem validade") from exc
    if expira <= now:
        raise OtaRejectedError("envelope vencido")
    manifesto = payload.get("manifest") or {}
    if manifesto.get("kind") not in KINDS:
        raise OtaRejectedError(f"tipo de pacote desconhecido: {manifesto.get('kind')!r}")
    if not manifesto.get("sha256") or not manifesto.get("files"):
        raise OtaRejectedError("manifesto sem hash ou sem arquivos")
    return payload


def _sha(dados: bytes) -> str:
    return hashlib.sha256(dados).hexdigest()


def extract_checked(dados: bytes, manifesto: dict, destino: Path) -> None:
    """Extrai conferindo **cada** arquivo contra o manifesto (nada a mais, nada a menos)."""
    esperados = {f["path"]: f for f in manifesto["files"]}
    vistos: set[str] = set()
    destino.mkdir(parents=True, exist_ok=True)
    raiz = destino.resolve()
    with tarfile.open(fileobj=io.BytesIO(dados), mode="r:*") as tar:
        for membro in tar.getmembers():
            if membro.isdir():
                continue
            nome = membro.name
            if not membro.isfile() or nome not in esperados:
                raise OtaRejectedError(f"arquivo fora do manifesto: {nome}")
            alvo = (destino / nome).resolve()
            if not alvo.is_relative_to(raiz):
                raise OtaRejectedError(f"caminho inseguro: {nome}")
            conteudo = tar.extractfile(membro).read()
            if _sha(conteudo) != esperados[nome]["sha256"]:
                raise OtaRejectedError(f"hash de {nome} não confere com o manifesto")
            alvo.parent.mkdir(parents=True, exist_ok=True)
            alvo.write_bytes(conteudo)
            vistos.add(nome)
    faltando = set(esperados) - vistos
    if faltando:
        raise OtaRejectedError(f"faltam arquivos do manifesto: {sorted(faltando)}")


@dataclass
class OtaReport:
    checked: int = 0
    applied: list[str] | None = None
    rejected: list[str] | None = None
    failed: list[str] | None = None
    deferred: str | None = None


class OtaManager:
    def __init__(
        self,
        store: LocalStore,
        client: DeviceClient | None,
        events: EventEngine,
        clock: Clock,
        *,
        public_key: str | None,
        device_id: str,
        accelerator: AiAccelerator,
        classification: ClassificationConfig,
        models_folder: Path | None = None,
        config_path: Path | None = None,
        chunk_size: int = 1024 * 1024,
        max_trial_boots: int = 2,
        mission_running: Callable[[], bool] = lambda: False,
        python: str = sys.executable,
        pacote: str = "valtexa_rover",
        tipos: tuple[str, ...] = KINDS,
    ) -> None:
        self.store = store
        self.client = client
        self.events = events
        self.clock = clock
        self.public_key = public_key
        self.device_id = device_id
        self.accelerator = accelerator
        self.classification = classification
        self.root = store.root / "ota"
        self.staging = self.root / "staging"
        self.slots = self.root / "slots"
        self.models_folder = Path(models_folder) if models_folder else self.root / "models"
        self.config_path = Path(config_path) if config_path else store.root / "rover.json"
        self.chunk_size = chunk_size
        self.max_trial_boots = max_trial_boots
        self.mission_running = mission_running
        self.python = python
        # A base da fazenda (VALTEXA-BASE) usa o mesmo OTA para o pacote dela:
        # ``pacote="valtexa_base", tipos=("runtime",)``.
        self.pacote = pacote
        self.tipos = tipos
        for pasta in (self.staging, self.slots, self.models_folder):
            pasta.mkdir(parents=True, exist_ok=True)

    # -- ponteiro (A/B) -------------------------------------------------------

    @property
    def pointer_path(self) -> Path:
        return self.root / "current.json"

    def current(self) -> dict:
        try:
            return json.loads(self.pointer_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _write_pointer(self, dados: dict) -> None:
        temp = self.pointer_path.with_suffix(".json.part")
        with open(temp, "w", encoding="utf-8") as fh:
            json.dump(dados, fh, ensure_ascii=False, indent=1)
            fh.flush()
            os.fsync(fh.fileno())
        temp.replace(self.pointer_path)  # troca atômica

    def active_path(self, kind: str) -> Path | None:
        """Pasta da versão ativa de ``runtime``/``edge_ai`` (o lançador põe no PYTHONPATH)."""
        item = self.current().get(kind)
        return Path(item["path"]) if item else None

    # -- boot -------------------------------------------------------------------

    def on_boot(self) -> list[str]:
        """Conta um boot de cada pacote em teste; passou do limite sem confirmar → volta."""
        ponteiro = self.current()
        revertidos = []
        for kind, item in list(ponteiro.items()):
            if item.get("state") != "teste":
                continue
            item["boots"] = int(item.get("boots", 0)) + 1
            if item["boots"] > self.max_trial_boots:
                self._rollback(ponteiro, kind, f"não confirmou em {self.max_trial_boots} boots")
                revertidos.append(kind)
        if any(i.get("state") == "teste" for i in self.current().values()):
            self._write_pointer(ponteiro)
        return revertidos

    def confirm_boot(self) -> list[str]:
        """O Runtime subiu inteiro: confirma o que estava em teste (vira o atual de vez)."""
        ponteiro = self.current()
        confirmados = []
        for kind, item in ponteiro.items():
            if item.get("state") == "teste":
                item["state"] = "confirmado"
                confirmados.append(kind)
                self._informar(item["deployment_id"], "aplicada",
                               f"{kind} {item['version']} confirmado no boot", item["version"])
                self.events.emit(EventType.OTA_APPLIED,
                                 {"kind": kind, "version": item["version"],
                                  "deployment_id": item["deployment_id"], "confirmed": True})
        if confirmados:
            self._write_pointer(ponteiro)
        return confirmados

    def _rollback(self, ponteiro: dict, kind: str, motivo: str) -> None:
        item = ponteiro[kind]
        anterior = item.get("previous")
        if kind == "config":
            backup = self.config_path.with_suffix(".json.anterior")
            if backup.exists():
                backup.replace(self.config_path)
            elif self.config_path.exists():
                self.config_path.unlink()
        if anterior:
            ponteiro[kind] = anterior
        else:
            del ponteiro[kind]
        self.events.emit(EventType.OTA_ROLLED_BACK,
                         {"kind": kind, "from": item.get("version"),
                          "to": anterior.get("version") if anterior else None,
                          "reason": motivo, "deployment_id": item.get("deployment_id")},
                         severity="critical")
        self._informar(item.get("deployment_id"), "revertida", motivo,
                       anterior.get("version") if anterior else None)

    # -- laço -------------------------------------------------------------------

    def step(self) -> OtaReport:
        """Um ciclo: informes pendentes, busca, baixa, confere, instala. Offline: nada."""
        rel = OtaReport(applied=[], rejected=[], failed=[])
        try:
            self._flush_reports()
            if not self.public_key:
                rel.deferred = "sem chave pública OTA (provisionamento antigo)"
                return rel
            resposta = self.client.request("GET", "/v1/device/ota")
            if resposta.status != 200:
                rel.deferred = f"HTTP {resposta.status}"
                return rel
            for item in resposta.json():
                rel.checked += 1
                self._handle(item, rel)
        except NetworkUnavailableError:
            rel.deferred = "offline"
        return rel

    def _handle(self, item: dict, rel: OtaReport) -> None:
        dep_id = item.get("deployment_id")
        processados = self.store.get("ota_processed", {}) or {}
        if dep_id in processados:
            return  # já decidido aqui: o informe está na fila ou já foi
        try:
            payload = verify_envelope(item.get("envelope") or {}, public_key=self.public_key,
                                      device_id=self.device_id, now=self.clock.now())
        except OtaRejectedError as exc:
            self._final(dep_id, "recusada", str(exc), EventType.OTA_REJECTED)
            rel.rejected.append(dep_id)
            return
        if payload["deployment_id"] != dep_id:
            self._final(dep_id, "recusada", "envelope de outra atualização",
                        EventType.OTA_REJECTED)
            rel.rejected.append(dep_id)
            return
        manifesto = payload["manifest"]
        if manifesto.get("kind") not in self.tipos:
            self._final(dep_id, "recusada",
                        f"pacote {manifesto.get('kind')} não se instala neste dispositivo",
                        EventType.OTA_REJECTED)
            rel.rejected.append(dep_id)
            return
        if self.mission_running():
            rel.deferred = "missão em execução: a instalação espera o fim"
            return
        try:
            dados = self._download(dep_id, manifesto)
            if dados is None:
                rel.deferred = "download incompleto"
                return
            self._informar(dep_id, "verificando", None)
            if len(dados) != manifesto["size_bytes"] or _sha(dados) != manifesto["sha256"]:
                (self.staging / f"{dep_id}.tar").unlink(missing_ok=True)
                raise OtaRejectedError("hash do pacote baixado não confere com o manifesto")
            self._install(dep_id, manifesto, dados)
        except OtaRejectedError as exc:
            self._final(dep_id, "recusada", str(exc), EventType.OTA_REJECTED)
            rel.rejected.append(dep_id)
        except OtaInstallError as exc:
            self._final(dep_id, "falhou", str(exc), EventType.OTA_FAILED)
            rel.failed.append(dep_id)
        else:
            rel.applied.append(dep_id)

    # -- download com retomada ---------------------------------------------------

    def _download(self, dep_id: str, manifesto: dict) -> bytes | None:
        pronto = self.staging / f"{dep_id}.tar"
        if pronto.exists():
            return pronto.read_bytes()
        parcial = self.staging / f"{dep_id}.tar.part"
        total = int(manifesto["size_bytes"])
        feito = parcial.stat().st_size if parcial.exists() else 0
        self._informar(dep_id, "baixando", f"{feito}/{total} bytes")
        with open(parcial, "ab") as fh:
            while feito < total:
                r = self.client.request(
                    "GET", f"/v1/device/ota/{dep_id}/package?offset={feito}"
                    f"&length={min(self.chunk_size, total - feito)}")
                if r.status != 200 or not r.body:
                    return None
                fh.write(r.body)
                fh.flush()
                os.fsync(fh.fileno())
                feito += len(r.body)
                self.store.put(f"ota_progress:{dep_id}", {"bytes": feito, "total": total})
        if feito != total:
            parcial.unlink()  # veio mais do que o manifesto diz: começa de novo
            return None
        parcial.replace(pronto)
        self.events.emit(EventType.OTA_DOWNLOADED, {"deployment_id": dep_id, "bytes": total})
        return pronto.read_bytes()

    # -- instalação lado a lado ------------------------------------------------

    def _install(self, dep_id: str, manifesto: dict, dados: bytes) -> None:
        kind, versao = manifesto["kind"], manifesto["version"]
        ponteiro = self.current()
        atual = ponteiro.get(kind)
        if atual and atual.get("version") == versao and atual.get("state") == "confirmado":
            self._final(dep_id, "aplicada", f"{kind} {versao} já instalado", None)
            return
        slot = self.slots / kind / versao
        if slot.exists():
            shutil.rmtree(slot)  # sobra de uma tentativa anterior: refaz do zero
        self._informar(dep_id, "instalando", str(slot))
        extract_checked(dados, manifesto, slot)
        self._informar(dep_id, "verificando_saude", None)
        try:
            self._health(kind, slot, manifesto)
        except OtaInstallError:
            shutil.rmtree(slot, ignore_errors=True)
            raise
        # Troca: o anterior fica guardado no ponteiro para o rollback.
        novo = {"version": versao, "path": str(slot), "deployment_id": dep_id,
                "installed_at": self.clock.now().isoformat(),
                "previous": {k: v for k, v in (atual or {}).items() if k != "previous"} or None}
        if kind == "model":
            self._activate_model(slot, manifesto)
            novo["state"] = "confirmado"
        elif kind == "config":
            if self.config_path.exists():
                shutil.copyfile(self.config_path, self.config_path.with_suffix(".json.anterior"))
            temp = self.config_path.with_suffix(".json.part")
            shutil.copyfile(slot / "rover.json", temp)
            temp.replace(self.config_path)
            novo["state"] = "teste"
            novo["boots"] = 0
        else:
            novo["state"] = "teste"
            novo["boots"] = 0
        ponteiro[kind] = novo
        self._write_pointer(ponteiro)
        self._mark_processed(dep_id)
        self.events.emit(EventType.OTA_INSTALLED,
                         {"kind": kind, "version": versao, "deployment_id": dep_id,
                          "state": novo["state"],
                          "previous": (atual or {}).get("version")})
        if novo["state"] == "confirmado":
            self._informar(dep_id, "aplicada", f"{kind} {versao} instalado e conferido", versao)
            self.events.emit(EventType.OTA_APPLIED, {"kind": kind, "version": versao,
                                                     "deployment_id": dep_id})
        (self.staging / f"{dep_id}.tar").unlink(missing_ok=True)

    def _health(self, kind: str, slot: Path, manifesto: dict) -> None:
        if kind == "model":
            self._health_model(slot, manifesto)
        elif kind == "config":
            arquivo = slot / "rover.json"
            if not arquivo.exists():
                raise OtaInstallError("pacote de configuração sem rover.json")
            try:
                RuntimeConfig.load(arquivo)
            except ConfigError as exc:
                raise OtaInstallError(f"configuração inválida: {exc}") from exc
        else:
            self._health_python(kind, slot, manifesto["version"])

    def _health_model(self, slot: Path, manifesto: dict) -> None:
        modelo = manifesto.get("model") or {}
        sha = modelo.get("artifact_sha256")
        arquivo = slot / f"{sha}.onnx"
        if not sha or not arquivo.exists():
            raise OtaInstallError("pacote de modelo sem o <sha256>.onnx do registry")
        from PIL import Image

        motor = EdgeAIEngine(self.accelerator, self.classification)
        pacote = ModelPackage(model_key=modelo.get("model_key") or "?",
                              version=modelo.get("version") or "?", sha256=sha, path=arquivo,
                              artifact_kind=modelo.get("artifact_kind") or "original",
                              dnaty_version=modelo.get("dnaty_version"))
        imagem = io.BytesIO()
        Image.new("RGB", (64, 64), (90, 140, 60)).save(imagem, "JPEG")
        try:
            motor.load(pacote)
            inferencia = motor.infer(imagem.getvalue())
        except Exception as exc:  # EdgeAIError ou o ONNX Runtime quebrando de outro jeito
            raise OtaInstallError(f"o modelo novo não carregou ou não inferiu: {exc}") from exc
        if not 0.0 <= inferencia.confidence <= 1.0:
            raise OtaInstallError("o modelo novo devolveu confiança fora de [0, 1]")

    def _health_python(self, kind: str, slot: Path, versao: str) -> None:
        pacote = slot / self.pacote
        if not (pacote / "__init__.py").exists():
            raise OtaInstallError(f"pacote {kind} sem {self.pacote}/__init__.py")
        if self.pacote == "valtexa_rover":
            atributo = "RUNTIME_VERSION" if kind == "runtime" else "EDGE_AI_VERSION"
        else:
            atributo = "__version__"
        codigo = (
            "import sys, compileall; sys.path.insert(0, sys.argv[1]);"
            "ok = compileall.compile_dir(sys.argv[1], quiet=2, force=True);"
            f"import {self.pacote} as m;"
            f"print(getattr(m, '{atributo}', '')); sys.exit(0 if ok else 3)"
        )
        try:
            saida = subprocess.run([self.python, "-S", "-c", codigo, str(slot)],
                                   capture_output=True, text=True, timeout=120, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise OtaInstallError(f"verificação de saúde não rodou: {exc}") from exc
        if saida.returncode != 0:
            raise OtaInstallError(f"{kind} novo não compila/importa: "
                                  f"{(saida.stderr or saida.stdout).strip()[-300:]}")
        if saida.stdout.strip() != versao:
            raise OtaInstallError(f"{kind} novo diz versão {saida.stdout.strip()!r}, "
                                  f"o manifesto diz {versao!r}")

    def _activate_model(self, slot: Path, manifesto: dict) -> None:
        """Põe o modelo na pasta de artefatos do Model Manager (ao lado dos que já há).

        Quem decide **qual** modelo roda continua sendo a designação do registry, e o
        Model Manager reconfere o hash antes de carregar.
        """
        sha = manifesto["model"]["artifact_sha256"]
        for arquivo in slot.iterdir():
            if arquivo.name.startswith(sha):
                temp = self.models_folder / (arquivo.name + ".part")
                shutil.copyfile(arquivo, temp)
                temp.replace(self.models_folder / arquivo.name)

    # -- informes ---------------------------------------------------------------

    def _final(self, dep_id: str | None, estado: str, detalhe: str,
               evento: EventType | None) -> None:
        if dep_id:
            self._mark_processed(dep_id)
            (self.staging / f"{dep_id}.tar").unlink(missing_ok=True)
            (self.staging / f"{dep_id}.tar.part").unlink(missing_ok=True)
        if evento is not None:
            self.events.emit(evento, {"deployment_id": dep_id, "reason": detalhe},
                             severity="critical" if evento is EventType.OTA_REJECTED
                             else "warning")
        self._informar(dep_id, estado, detalhe)

    def _mark_processed(self, dep_id: str) -> None:
        processados = self.store.get("ota_processed", {}) or {}
        processados[dep_id] = self.clock.now().isoformat()
        self.store.put("ota_processed", processados)

    def _informar(self, dep_id: str | None, estado: str, detalhe: str | None,
                  versao: str | None = None) -> None:
        if not dep_id:
            return
        fila = self.store.get("ota_reports", []) or []
        fila.append({"deployment_id": dep_id, "state": estado, "detail": detalhe,
                     "version_running": versao})
        self.store.put("ota_reports", fila)
        with contextlib.suppress(NetworkUnavailableError):  # offline: sai no próximo passo
            self._flush_reports()

    def _flush_reports(self) -> None:
        if self.client is None:
            return  # no lançador (antes do Runtime subir) não há rede: sai depois
        fila = self.store.get("ota_reports", []) or []
        while fila:
            item = fila[0]
            r = self.client.request(
                "POST", f"/v1/device/ota/{item['deployment_id']}/status",
                json_body={k: item[k] for k in ("state", "detail", "version_running")})
            if r.status >= 500:
                break  # tenta de novo depois
            fila.pop(0)  # 2xx, ou 4xx definitivo (transição já registrada / inválida)
            self.store.put("ota_reports", fila)


__all__ = ["KINDS", "OtaInstallError", "OtaManager", "OtaRejectedError", "canonical_ota",
           "extract_checked", "verify_envelope"]
