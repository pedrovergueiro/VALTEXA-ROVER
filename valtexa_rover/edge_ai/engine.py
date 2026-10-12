"""Edge AI Engine: pré-processamento → inferência ONNX → pós-processamento → categoria.

Brief, seções 11, 14 e 22. Roda **offline**, no computador do Rover, com o
modelo que o Model Manager carregou (sempre um modelo do registry, com hash
conferido — ver :mod:`valtexa_rover.runtime.model_manager`).

O pré-processamento é o do treino, copiado byte a byte do ``ia/``
(:mod:`valtexa_rover.edge_ai.vendor.preprocess`): decodifica sem EXIF,
redimensiona como o ``tf.image.resize`` e escala para ``[-1, 1]``.

Categorias (seção 14): cada rótulo do modelo tem uma categoria — NORMAL,
POSSIVEL_PRAGA, POSSIVEL_DOENCA ou ANOMALIA. O pacote do modelo pode trazer
``<modelo>.categories.json`` (rótulo → categoria), que é o recomendado; sem ele,
:func:`categoria_padrao` deduz pelo nome do rótulo. Por cima disso:

* confiança abaixo de ``anomaly_below`` → **ANOMALIA** (nenhuma classe do modelo
  foi reconhecida: é o que a seção 14 chama de anomalia — o desconhecido);
* abaixo de ``low_confidence_below`` → mantém a categoria e marca
  ``baixa_confianca`` (a retenção preserva o original, seção 16).

Os limiares são configuração (``classificacao`` no ``rover.json``) e **precisam
ser calibrados com o modelo real** — aqui não há pesos reais; os testes usam um
modelo ONNX minúsculo gerado na hora.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from valtexa_rover.config import Category, ClassificationConfig
from valtexa_rover.edge_ai.perfil import EdgeProfile, ProfileError
from valtexa_rover.edge_ai.vendor.heads import DiagnosisHead
from valtexa_rover.edge_ai.vendor.onnx_runtime import OnnxModel
from valtexa_rover.edge_ai.vendor.preprocess import (
    InvalidImageError,
    load_image_from_bytes,
    preprocess_batch,
    preprocess_image,
)
from valtexa_rover.edge_ai.vendor.rover_geometry import ZONES, plan_tiles
from valtexa_rover.hal.interfaces import AiAccelerator


class EdgeAIError(RuntimeError):
    """Modelo ausente, inválido ou incompatível com os rótulos."""


PALAVRAS_PRAGA = ("miner", "mineiro", "praga", "pest", "lagarta", "broca", "acaro", "ácaro",
                  "mite", "insect", "inseto", "cochonilha", "trips", "pulgao", "pulgão")
PALAVRAS_NORMAL = ("healthy", "saudavel", "saudável", "normal", "sadia")
PALAVRAS_ANOMALIA = ("anomal", "suspect", "desconhecid", "unknown", "background", "fundo")


def categoria_padrao(rotulo: str) -> Category:
    """Dedução pelo nome do rótulo, quando o modelo não traz ``categories.json``."""
    nome = rotulo.lower()
    if any(p in nome for p in PALAVRAS_NORMAL):
        return Category.NORMAL
    if any(p in nome for p in PALAVRAS_ANOMALIA):
        return Category.ANOMALIA
    if any(p in nome for p in PALAVRAS_PRAGA):
        return Category.POSSIVEL_PRAGA
    return Category.POSSIVEL_DOENCA


@dataclass(frozen=True, slots=True)
class ModelPackage:
    """Um modelo pronto para a Edge AI: o grafo e o que o registry diz dele."""

    model_key: str
    version: str
    sha256: str
    path: Path
    artifact_kind: str = "original"
    """``original`` ou ``otimizado`` (dNATY)."""
    dnaty_version: str | None = None
    target_hardware: str | None = None

    @property
    def labels_path(self) -> Path:
        return self.path.with_suffix(".labels.json")

    @property
    def categories_path(self) -> Path:
        return self.path.with_suffix(".categories.json")

    @property
    def profile_path(self) -> Path:
        """Perfil da Edge AI (recorte e limiares medidos) — ver :mod:`.perfil`."""
        return self.path.with_suffix(".edge.json")

    @property
    def head_path(self) -> Path:
        """Cabeça sobre o embedding da torre de visão (``training.fit_heads`` no ``ia/``)."""
        return self.path.with_suffix(".head.npz")

    @property
    def encoder_path(self) -> Path:
        """A torre de visão do SigLIP 2 em ONNX (``framing_vision.onnx`` no ``ia/``)."""
        return self.path.with_suffix(".encoder.onnx")


@dataclass(frozen=True, slots=True)
class TileResult:
    """Um recorte de tamanho físico fixo do quadro, com o que o modelo viu nele."""

    box: tuple[int, int, int, int]
    height_mm: float
    zones: tuple[str, ...]
    probabilities: tuple[float, ...]
    """Na ordem dos rótulos do modelo (``EdgeAIEngine.labels``)."""


@dataclass(frozen=True, slots=True)
class Inference:
    label: str
    confidence: float
    category: Category
    low_confidence: bool
    top: list[tuple[str, float]]
    latency_ms: float
    input_size: int
    reason: str = ""
    tiles: tuple[TileResult, ...] = ()
    """Vazio quando o modelo classifica o quadro inteiro (sem perfil)."""
    tile_mm: float | None = None
    event_tiles: int = 0
    """Recortes com doença/praga acima do limiar do perfil (na regra por evidência: os de
    evidência a favor da doença do quadro)."""
    evidence: float | None = None
    """Escore de evidência do quadro (regra por evidência do perfil)."""
    distance_mm: float | None = None
    distance_measured: bool = False
    hfov_deg: float | None = None
    """O campo de visão com que o quadro foi cortado (o efetivo da lente corrigida, ou o
    nominal do perfil)."""


@dataclass(slots=True)
class EngineStats:
    """Contadores que o Health Monitor e a telemetria leem."""

    inferences: int = 0
    errors: int = 0
    total_latency_ms: float = 0.0
    latencies: list[float] = field(default_factory=list)

    def add(self, latency_ms: float) -> None:
        self.inferences += 1
        self.total_latency_ms += latency_ms
        self.latencies.append(latency_ms)
        del self.latencies[:-500]

    @property
    def mean_latency_ms(self) -> float | None:
        return self.total_latency_ms / self.inferences if self.inferences else None

    @property
    def fps(self) -> float | None:
        """Quadros por segundo que a inferência sustenta (pela latência média recente)."""
        if not self.latencies:
            return None
        media = sum(self.latencies) / len(self.latencies)
        return 1000.0 / media if media > 0 else None


class EdgeAIEngine:
    def __init__(self, accelerator: AiAccelerator, config: ClassificationConfig) -> None:
        self.accelerator = accelerator
        self.config = config
        self.package: ModelPackage | None = None
        self.labels: tuple[str, ...] = ()
        self.categories: dict[str, Category] = {}
        self.stats = EngineStats()
        self.profile: EdgeProfile | None = None
        self.fixed_batch = False
        """Lote de recortes sempre do tamanho do perfil (TensorRT recompila a cada forma nova)."""
        self._model: OnnxModel | None = None
        self._size = 0
        self.head: DiagnosisHead | None = None
        self._encoder: OnnxModel | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self, package: ModelPackage) -> None:
        """Carrega o grafo e os rótulos. Não confere hash: isso é do Model Manager."""
        if not package.path.exists():
            raise EdgeAIError(f"Modelo {package.model_key}: arquivo {package.path} não existe.")
        if not package.labels_path.exists():
            raise EdgeAIError(
                f"Modelo {package.model_key}: faltam os rótulos ({package.labels_path.name})."
            )
        labels = json.loads(package.labels_path.read_text(encoding="utf-8"))
        if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
            raise EdgeAIError(f"{package.labels_path.name} deveria ser uma lista de strings.")
        categorias = {rotulo: categoria_padrao(rotulo) for rotulo in labels}
        if package.categories_path.exists():
            bruto = json.loads(package.categories_path.read_text(encoding="utf-8"))
            try:
                categorias.update({k: Category(v) for k, v in bruto.items() if k in categorias})
            except ValueError as exc:
                raise EdgeAIError(f"{package.categories_path.name}: {exc}") from exc
        try:
            model = OnnxModel(package.path, providers=self.accelerator.providers())
        except Exception as exc:  # o ONNX Runtime levanta tipos próprios
            raise EdgeAIError(f"ONNX Runtime recusou {package.path.name}: {exc}") from exc

        shape = model.input_shape
        if len(shape) != 4 or shape[1] != shape[2] or shape[3] != 3 or not shape[1]:
            raise EdgeAIError(f"Entrada esperada (N, S, S, 3); o modelo declara {shape}.")
        saidas = model.output_shape[-1]
        if saidas is not None and saidas != len(labels):
            raise EdgeAIError(
                f"O modelo tem {saidas} saídas e {package.labels_path.name} tem {len(labels)}."
            )
        perfil = None
        if package.profile_path.exists():
            try:
                perfil = EdgeProfile.load(package.profile_path)
            except ProfileError as exc:
                raise EdgeAIError(f"{package.profile_path.name}: {exc}") from exc
            if perfil.healthy_label not in labels:
                raise EdgeAIError(
                    f"{package.profile_path.name}: rótulo sadio {perfil.healthy_label!r} não está "
                    f"em {package.labels_path.name}."
                )
            if perfil.evidence is not None:
                faltam = sorted(set(perfil.evidence.llr_tables) - set(labels))
                if faltam:
                    raise EdgeAIError(
                        f"{package.profile_path.name}: evidência para rótulos que o modelo não "
                        f"tem: {', '.join(faltam)}."
                    )
        cabeca, torre = (None, None)
        if perfil is not None and perfil.head is not None:
            cabeca, torre = self._load_head(package, perfil, tuple(labels))
        self._model = model
        self._size = int(shape[1])
        self.package = package
        self.labels = tuple(labels)
        self.categories = categorias
        self.profile = perfil
        self.head, self._encoder = cabeca, torre
        self.fixed_batch = "TensorrtExecutionProvider" in getattr(model, "providers", [])

    def _load_head(self, package: ModelPackage, perfil: EdgeProfile,
                   labels: tuple[str, ...]) -> tuple[DiagnosisHead, OnnxModel]:
        """A cabeça e a torre que o perfil pede, conferidas antes de valerem."""
        spec = perfil.head
        assert spec is not None
        for arquivo in (package.head_path, package.encoder_path):
            if not arquivo.exists():
                raise EdgeAIError(f"O perfil pede a cabeça, mas falta {arquivo.name}.")
        sha = hashlib.sha256()
        with package.encoder_path.open("rb") as fh:
            for bloco in iter(lambda: fh.read(1 << 20), b""):
                sha.update(bloco)
        if sha.hexdigest() != spec.encoder_sha256:
            raise EdgeAIError(
                f"{package.encoder_path.name}: SHA-256 {sha.hexdigest()[:12]}… não é o da torre "
                f"medida no perfil ({spec.encoder_sha256[:12]}…)."
            )
        try:
            cabeca = DiagnosisHead.load(package.head_path)
        except ValueError as exc:
            raise EdgeAIError(str(exc)) from exc
        if cabeca.class_names != labels:
            raise EdgeAIError(
                f"{package.head_path.name}: classes {list(cabeca.class_names)} diferem dos "
                f"rótulos do modelo {list(labels)}."
            )
        if cabeca.encoder != spec.encoder_model_id:
            raise EdgeAIError(
                f"{package.head_path.name} foi treinada sobre {cabeca.encoder}, e o perfil "
                f"diz {spec.encoder_model_id}."
            )
        try:
            torre = OnnxModel(package.encoder_path, providers=self.accelerator.providers())
        except Exception as exc:  # o ONNX Runtime levanta tipos próprios
            raise EdgeAIError(f"ONNX Runtime recusou {package.encoder_path.name}: {exc}") from exc
        if torre.output_shape[-1] not in (None, cabeca.mlp.input_size):
            raise EdgeAIError(
                f"A torre dá embedding de {torre.output_shape[-1]}; a cabeça espera "
                f"{cabeca.mlp.input_size}."
            )
        return cabeca, torre

    def _run_batched(self, rede: OnnxModel, lote: np.ndarray, tamanho: int) -> np.ndarray:
        """Roda em lotes de ``tamanho``, completando o último sob TensorRT."""
        saidas = []
        for i in range(0, len(lote), tamanho):
            parte = lote[i:i + tamanho]
            n = len(parte)
            if self.fixed_batch and n < tamanho:  # completa repetindo o último
                parte = np.concatenate([parte, np.repeat(parte[-1:], tamanho - n, 0)])
            saidas.append(np.asarray(rede(parte), dtype=np.float64)[:n])
        return np.concatenate(saidas)

    def _tile_probabilities(self, pedacos: list) -> np.ndarray:
        """Probabilidades de cada recorte: a CNN, ou a cabeça sobre a torre de visão."""
        perfil = self.profile
        assert perfil is not None and self._model is not None
        cabeca = self.head
        cnn = None
        if cabeca is None or cabeca.mix.get("cnn", 0.0) > 0:
            lote = np.concatenate([preprocess_image(p, self._size) for p in pedacos])
            cnn = np.vstack([_probabilidades(linha)
                             for linha in self._run_batched(self._model, lote, perfil.tile_batch)])
        if cabeca is None:
            return cnn
        assert self._encoder is not None and perfil.head is not None
        # O redimensionamento do Pillow, como o processador do SigLIP (o mesmo de
        # ``app.framing.onnx_vision_encoder`` no ``ia/``); a torre recebe NCHW.
        lote = preprocess_batch([p.convert("RGB") for p in pedacos], perfil.head.input_size,
                                resampling="pillow").transpose(0, 3, 1, 2)
        embeddings = self._run_batched(self._encoder, lote, perfil.tile_batch)
        if cnn is None:  # mistura sem a CNN: o valor não entra na conta
            cnn = np.full((len(pedacos), len(self.labels)), 1.0 / len(self.labels))
        return cabeca.combine(cnn, embeddings)

    def infer(
        self,
        image_bytes: bytes,
        distance_mm: float | None = None,
        camera_height_mm: float | None = None,
        hfov_deg: float | None = None,
    ) -> Inference:
        """Classifica um quadro. Com perfil, recorta em tamanho físico fixo (ver :mod:`.perfil`).

        ``distance_mm`` vem da profundidade da câmera; ``camera_height_mm``, do andar do
        mastro. Sem eles valem a distância padrão do perfil e 1.000 mm.
        """
        if self._model is None or self.package is None:
            raise EdgeAIError("Nenhum modelo carregado.")
        inicio = time.perf_counter()
        try:
            image = load_image_from_bytes(image_bytes)
        except InvalidImageError:
            self.stats.errors += 1
            raise
        if self.profile is not None:
            return self._infer_tiles(image, inicio, distance_mm, camera_height_mm, hfov_deg)
        batch = preprocess_image(image, self._size)
        saida = np.asarray(self._model(batch), dtype=np.float64)[0]
        probs = _probabilidades(saida)
        latency = (time.perf_counter() - inicio) * 1000
        self.stats.add(latency)

        ordem = np.argsort(probs)[::-1]
        top = [(self.labels[i], round(float(probs[i]), 6)) for i in ordem[:3]]
        rotulo, confianca = top[0][0], float(probs[ordem[0]])
        categoria = self.categories[rotulo]
        baixa = confianca < self.config.low_confidence_below
        motivo = ""
        if confianca < self.config.anomaly_below:
            categoria = Category.ANOMALIA
            motivo = (
                f"confiança {confianca:.2f} abaixo de {self.config.anomaly_below:.2f}: "
                "nenhuma classe reconhecida"
            )
        elif baixa:
            motivo = f"confiança {confianca:.2f} abaixo de {self.config.low_confidence_below:.2f}"
        return Inference(rotulo, round(confianca, 6), categoria, baixa, top, round(latency, 3),
                         self._size, motivo)

    def _infer_tiles(self, image, inicio: float, distance_mm, camera_height_mm,
                     hfov_deg: float | None = None) -> Inference:
        perfil = self.profile
        assert perfil is not None and self._model is not None
        medida = distance_mm is not None and distance_mm > 0
        distancia = float(distance_mm) if medida else perfil.default_distance_mm
        altura = float(camera_height_mm) if camera_height_mm is not None else 1000.0
        if hfov_deg is not None:
            # Lente corrigida: câmera ideal de pixel quadrado; o vertical sai da proporção.
            hfov = float(hfov_deg)
            meia = math.tan(math.radians(hfov) / 2) * image.size[1] / image.size[0]
            vfov = math.degrees(2 * math.atan(meia))
        else:
            hfov, vfov = perfil.hfov_deg, perfil.vfov_deg
        try:
            recortes = plan_tiles(
                image.size, distancia, altura, crop=perfil.crop if perfil.crop in ZONES else None,
                tile_mm=perfil.tile_mm, overlap=perfil.overlap,
                hfov_deg=hfov, vfov_deg=vfov,
            )
        except ValueError as exc:
            # Planta longe demais para a resolução: o modelo não lê. Guarda o original
            # (baixa confiança) para a nuvem decidir, em vez de chamar de sadio.
            latency = (time.perf_counter() - inicio) * 1000
            self.stats.add(latency)
            return Inference(perfil.healthy_label, 0.0, Category.NORMAL, True, [],
                             round(latency, 3), self._size, f"sem recorte: {exc}",
                             tile_mm=perfil.tile_mm, distance_mm=distancia,
                             distance_measured=medida, hfov_deg=hfov)
        if not recortes:
            latency = (time.perf_counter() - inicio) * 1000
            self.stats.add(latency)
            return Inference(perfil.healthy_label, 0.0, Category.NORMAL, True, [],
                             round(latency, 3), self._size, "quadro sem recorte acima do chão",
                             tile_mm=perfil.tile_mm, distance_mm=distancia,
                             distance_measured=medida, hfov_deg=hfov)
        probs = self._tile_probabilities([image.crop(t.box) for t in recortes])
        latency = (time.perf_counter() - inicio) * 1000
        self.stats.add(latency)

        sadio = self.labels.index(perfil.healthy_label)
        doenca = np.delete(probs, sadio, axis=1)
        rotulos_doenca = [r for i, r in enumerate(self.labels) if i != sadio]
        melhor_doenca = doenca.max(axis=1)
        achados = melhor_doenca >= perfil.event_threshold
        n_achados = int(achados.sum())
        topo = int(np.argmax(melhor_doenca))
        confianca_media = float(probs.max(axis=1).mean())
        escore = None

        regra = perfil.evidence
        if regra is not None:
            escores = {d: regra.score(d, [float(x) for x in probs[:, self.labels.index(d)]])
                       for d in regra.llr_tables}
            rotulo_ev = max(escores, key=escores.get)
            escore = escores[rotulo_ev]
            coluna = probs[:, self.labels.index(rotulo_ev)]
            n_achados = sum(regra.tile_llr(rotulo_ev, float(p)) > 0 for p in coluna)
            if escore > regra.event_evidence:
                rotulo = rotulo_ev
                categoria = self.categories[rotulo]
                confianca = float(coluna.max())
                baixa = False
                motivo = (f"evidência de {rotulo} {escore:.2f} > {regra.event_evidence:.2f} "
                          f"({n_achados} de {len(recortes)} recortes a favor)")
            else:
                rotulo = perfil.healthy_label
                categoria = Category.NORMAL
                confianca = float(probs[:, sadio].mean())
                baixa = (regra.near_miss_evidence is not None
                         and escore > regra.near_miss_evidence)
                motivo = (f"evidência de {rotulo_ev} {escore:.2f}, abaixo do evento "
                          f"({regra.event_evidence:.2f})" if baixa else "")
        elif n_achados >= perfil.min_event_tiles:
            # A doença do quadro: a de maior probabilidade somada nos recortes achados.
            soma = doenca[achados].sum(axis=0)
            rotulo = rotulos_doenca[int(np.argmax(soma))]
            categoria = self.categories[rotulo]
            confianca = float(melhor_doenca[achados].max())
            baixa = False
            motivo = (f"{n_achados} de {len(recortes)} recortes com {rotulo} ≥ "
                      f"{perfil.event_threshold:.2f}")
        else:
            rotulo = perfil.healthy_label
            categoria = Category.NORMAL
            confianca = float(probs[:, sadio].mean())
            baixa = (perfil.near_miss_threshold is not None
                     and float(melhor_doenca[topo]) >= perfil.near_miss_threshold)
            motivo = (f"recorte com {rotulos_doenca[int(np.argmax(doenca[topo]))]} "
                      f"{float(melhor_doenca[topo]):.2f}, abaixo do evento" if baixa else "")
        if (perfil.anomaly_mean_confidence_below is not None
                and confianca_media < perfil.anomaly_mean_confidence_below
                and categoria is Category.NORMAL):
            categoria = Category.ANOMALIA
            motivo = (f"confiança média {confianca_media:.2f} nos recortes, abaixo de "
                      f"{perfil.anomaly_mean_confidence_below:.2f}: o modelo não reconhece")

        media = probs.mean(axis=0)
        ordem = np.argsort(media)[::-1]
        top = [(self.labels[i], round(float(media[i]), 6)) for i in ordem[:3]]
        tiles = tuple(
            TileResult(tuple(int(v) for v in t.box), float(t.center_height_mm), tuple(t.zones),
                       tuple(round(float(p), 5) for p in linha))
            for t, linha in zip(recortes, probs, strict=True)
        )
        return Inference(rotulo, round(confianca, 6), categoria, baixa, top, round(latency, 3),
                         self._size, motivo, tiles=tiles, tile_mm=perfil.tile_mm,
                         event_tiles=int(n_achados), distance_mm=distancia,
                         distance_measured=medida, hfov_deg=hfov,
                         evidence=None if escore is None else round(escore, 6))


def _probabilidades(saida: np.ndarray) -> np.ndarray:
    """Os modelos da ValTexa saem com softmax; se vier *logit*, normaliza aqui."""
    if np.all(saida >= 0) and abs(float(saida.sum()) - 1.0) < 1e-3:
        return saida
    z = saida - saida.max()
    e = np.exp(z)
    return e / e.sum()
