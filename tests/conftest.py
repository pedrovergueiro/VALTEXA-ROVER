"""Fixtures do pacote do Rover.

**Não há pesos reais aqui.** O modelo dos testes é um grafo ONNX minúsculo,
gerado na hora com ``onnx.helper``: média dos canais RGB → matriz 3x4 →
softmax. Ele "reconhece" a cor dominante (verde = Healthy, vermelho = Rust,
azul = Leaf_Miner_Damage) e fica em dúvida no cinza — o suficiente para exercer
pré-processamento, ONNX Runtime, pós-processamento e categorias de verdade.
O desempenho de um modelo real é outra conversa: "a verificar com o modelo real".

A nuvem dos testes deste pacote é :class:`FakeCloud`, um **dublê** em memória
com as mesmas regras do SaaS (idempotência por ID e hash, chunks, retomada). O
teste contra o SaaS de verdade está em ``saas/tests/test_rover_integracao.py``.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from valtexa_rover._compat import UTC
from valtexa_rover.clock import ManualClock
from valtexa_rover.config import RuntimeConfig, SyncConfig
from valtexa_rover.hal.interfaces import (  # noqa: F401 (testes importam daqui)
    Hardware,
    HttpResponse,
)
from valtexa_rover.hal.local import LocalDiskStorage, OnnxCpuAccelerator
from valtexa_rover.hal.simulated import (
    FlakyNetwork,
    FolderCamera,
    SimulatedImu,
    SimulatedSensors,
    TrajectoryGps,
)
from valtexa_rover.runtime.client import DeviceIdentity, generate_keypair
from valtexa_rover.runtime.core import RoverCore
from valtexa_rover.runtime.model_manager import FolderArtifactSource
from valtexa_rover.sim.nuvem import NuvemEmMemoria, _json  # noqa: F401

LABELS = ["Healthy", "Rust", "Leaf_Miner_Damage", "Unknown_Background"]
COLORS = {
    "verde": (0, 255, 0),
    "vermelho": (255, 0, 0),
    "azul": (0, 0, 255),
    "cinza": (128, 128, 128),
    "duvida": (185, 191, 0),  # quase empate Healthy/Rust → baixa confiança
}


def build_tiny_onnx(path: Path, size: int = 32, k: float = 4.0) -> str:
    """Grava o modelo minúsculo e os rótulos ao lado; devolve o SHA-256 do ``.onnx``."""
    import onnx
    from onnx import TensorProto, helper

    pesos = np.array(
        [[-k, k, -k, 0.0], [k, -k, -k, 0.0], [-k, -k, k, 0.0]], dtype=np.float32
    )
    grafo = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["input"], ["media"], axes=[1, 2], keepdims=0),
            helper.make_node("MatMul", ["media", "W"], ["logits"]),
            helper.make_node("Softmax", ["logits"], ["probs"], axis=1),
        ],
        "valtexa_teste",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", size, size, 3])],
        [helper.make_tensor_value_info("probs", TensorProto.FLOAT, ["batch", len(LABELS)])],
        initializer=[helper.make_tensor("W", TensorProto.FLOAT, pesos.shape, pesos.flatten())],
    )
    modelo = helper.make_model(grafo, opset_imports=[helper.make_opsetid("", 17)])
    modelo.ir_version = 8
    onnx.checker.check_model(modelo)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(modelo.SerializeToString())
    path.with_suffix(".labels.json").write_text(json.dumps(LABELS))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_bytes(color: str, size: tuple[int, int] = (160, 120), noise: int = 0,
                seed: int = 0) -> bytes:
    rgb = np.zeros((size[1], size[0], 3), dtype=np.int16) + np.array(COLORS[color])
    if noise:
        rgb += np.random.default_rng(seed).integers(-noise, noise + 1, rgb.shape)
    imagem = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    saida = io.BytesIO()
    imagem.save(saida, format="JPEG", quality=92)
    return saida.getvalue()


def write_images(folder: Path, sequence: list[str], **kwargs) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for i, cor in enumerate(sequence):
        (folder / f"{i:05d}_{cor}.jpg").write_bytes(image_bytes(cor, seed=i, **kwargs))
    return folder


FakeCloud = NuvemEmMemoria
"""DUBLÊ da nuvem para os testes do pacote — mora em ``valtexa_rover.sim.nuvem``."""


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock(datetime(2026, 10, 4, 12, 0, tzinfo=UTC))


@pytest.fixture
def model_repo(tmp_path: Path) -> tuple[Path, str]:
    """Pasta de artefatos com o modelo minúsculo, nomeado pelo hash (como o OTA entrega)."""
    repo = tmp_path / "artefatos"
    temp = tmp_path / "build" / "modelo.onnx"
    digest = build_tiny_onnx(temp)
    repo.mkdir()
    (repo / f"{digest}.onnx").write_bytes(temp.read_bytes())
    (repo / f"{digest}.labels.json").write_text(json.dumps(LABELS))
    return repo, digest


def designation(field_id: str, sha: str | None, **extra) -> dict:
    return {"field_id": field_id, "field_name": "Talhão 1", "crop": "coffee",
            "model_key": "coffee-foliar", "version": "2.4.1", "state": "producao",
            "artifact_uri": f"s3://modelos/{sha}.onnx", "artifact_sha256": sha,
            "artifact_kind": "otimizado", "dnaty_version": "2.2.1",
            "target_hardware": "Rover Edge V1", "reason": None, **extra}


class Rig:
    """Um Rover simulado completo, montado sobre a HAL simulada."""

    def __init__(self, tmp_path: Path, clock: ManualClock, model_repo, sequence: list[str],
                 *, config: RuntimeConfig | None = None, data_dir: Path | None = None,
                 cloud: FakeCloud | None = None, **core_kwargs) -> None:
        privada, publica = generate_keypair()
        self.identity = DeviceIdentity("dev-1", "dk_teste", privada)
        repo, self.sha = model_repo
        self.cloud = cloud or FakeCloud(publica, [designation("talhao-1", self.sha)])
        self.network = FlakyNetwork(self.cloud)
        self.images = write_images(tmp_path / "imagens", sequence)
        self.gps = TrajectoryGps([(-45.0, -21.0), (-45.0, -20.999)], clock, speed_ms=1.0)
        self.hardware = Hardware(
            cameras=[FolderCamera("cam-0", self.images, clock, loop=False)],
            gps=self.gps,
            imu=SimulatedImu(clock, self.gps),
            sensors=SimulatedSensors(clock),
            storage=LocalDiskStorage(tmp_path / "dados"),
            network=self.network,
            accelerator=OnnxCpuAccelerator(),
            hardware_version="Rover Edge V1 (simulado)",
        )
        self.data_dir = data_dir or tmp_path / "dados"
        self.clock = clock
        self.repo = repo
        self.config = config or RuntimeConfig(sync=SyncConfig(chunk_size=4096))
        self.core_kwargs = {"telemetry_interval_s": 5.0, "seed": 1, **core_kwargs}
        self.core = self.new_core()

    def new_core(self) -> RoverCore:
        core = RoverCore(self.hardware, self.data_dir, self.identity, config=self.config,
                         clock=self.clock, artifact_source=FolderArtifactSource(self.repo),
                         **self.core_kwargs)
        core.boot()
        return core

    def run(self, frames: int) -> list:
        saida = []
        for _ in range(frames):
            saida.append(self.core.step())
            self.clock.advance(1.0)
        return saida


@pytest.fixture
def rig_factory(tmp_path, clock, model_repo):
    def fabrica(sequence: list[str], **kwargs) -> Rig:
        return Rig(tmp_path, clock, model_repo, sequence, **kwargs)
    return fabrica


def angle_ok(value: float) -> bool:
    return 0 <= value < 360 and not math.isnan(value)
