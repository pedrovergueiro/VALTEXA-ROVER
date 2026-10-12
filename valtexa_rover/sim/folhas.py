"""SIMULAÇÃO com imagem real: a câmera que monta o quadro com fotos de folha de verdade.

A :class:`~valtexa_rover.sim.CameraDeCena` pinta a planta de uma cor — serve para o fluxo,
não para a IA. Esta câmera serve para a IA: cada recorte que a Edge AI vai ler
(:func:`~valtexa_rover.edge_ai.vendor.rover_geometry.plan_tiles`, com a mesma distância,
altura e campo de visão do quadro) recebe uma foto de folha de uma pasta
``<pasta>/<Classe>/``. Numa planta sadia, toda folha é da classe sadia; numa doente, cada
recorte é uma folha doente com probabilidade ``incidencia`` (a fração de folhas com a
doença, como o agrônomo mede) e sadia no resto.

O que é real: as fotos (use um acervo que o modelo **não** viu no treino), o modelo, o
recorte e a decisão. O que é simulado: a montagem — no cafezal o recorte mostra folhas
embaraçadas, galho, fruto e sombra; aqui é uma folha em close por recorte. É o mesmo
método da calibração do café em ``VALTEXA-B/training/fit_edge.py mosaico``.

**Um recorte por quadro, por padrão.** O perfil sobrepõe os recortes em 25%: num quadro
de várias fotos, cada recorte leva a emenda com as vizinhas, e a IA lê a emenda como
lesão — medido no café: folhas sadias soltas dão P(sadia) de 0,95 a 0,96; as mesmas
montadas em 4 colunas e 3 linhas, de 0,06 a 0,95, e todo quadro vira evento. Com
``colunas = linhas = 1`` o recorte é a foto inteira. Mais de um por quadro só com um
acervo de fotos de planta.
"""

from __future__ import annotations

import hashlib
import io
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from valtexa_rover.clock import Clock
from valtexa_rover.edge_ai.vendor.rover_geometry import ZONES, plan_tiles
from valtexa_rover.hal.interfaces import DeviceInfo, Frame, HardwareUnavailableError
from valtexa_rover.hal.simulated import TrajectoryGps
from valtexa_rover.runtime.model_manager import EXTRAS

CLASSE_DA_CONDICAO = {"sadia": "Healthy", "doenca": "Rust", "praga": "Leaf_Miner_Damage"}
"""Condição da planta simulada → pasta de fotos (classes do modelo de café)."""


def instalar_modelo(origem: Path | str, pasta: Path | str) -> str:
    """Copia ``<modelo>.onnx`` e os arquivos ao lado para ``<sha256>.*`` em ``pasta``.

    É o formato da :class:`~valtexa_rover.runtime.model_manager.FolderArtifactSource`.
    Devolve o SHA-256 do ``.onnx`` — o que o registry precisa ter publicado.
    """
    origem, pasta = Path(origem), Path(pasta)
    sha = hashlib.sha256(origem.read_bytes()).hexdigest()
    pasta.mkdir(parents=True, exist_ok=True)
    for sufixo in (".onnx", *EXTRAS):
        fonte = origem.with_suffix(sufixo)
        destino = pasta / f"{sha}{sufixo}"
        if fonte.exists() and not destino.exists():
            shutil.copyfile(fonte, destino)
    return sha


class CameraDeFolhas:
    """SIMULAÇÃO: fotografa a planta mais próxima do GPS com folhas reais em cada recorte."""

    def __init__(self, camera_id: str, campo, gps: TrajectoryGps, clock: Clock,
                 folhas: Path | str, *, lado: str = "left", altura_mm: float = 1000.0,
                 distancia_mm: float = 270.0, colunas: int = 1, linhas: int = 1,
                 tile_mm: float = 150.0, overlap: float = 0.25, px_por_recorte: int = 224,
                 incidencia: float = 0.5, crop: str = "coffee", seed: int = 0) -> None:
        self.camera_id = camera_id
        self.campo = campo
        self.gps = gps
        self.clock = clock
        self.lado = lado
        self.altura_mm = altura_mm
        self.distancia_mm = distancia_mm
        self.incidencia = incidencia
        self.rng = np.random.default_rng(seed)
        self.fotos = {c: sorted(p for p in (Path(folhas) / c).iterdir() if p.is_file())
                      for c in set(CLASSE_DA_CONDICAO.values())
                      if (Path(folhas) / c).is_dir()}
        if not self.fotos.get("Healthy"):
            raise ValueError(f"{folhas}: sem fotos em Healthy/")
        # O quadro cobre colunas por linhas recortes com a sobreposição do perfil; o campo de
        # visão sai do tamanho em mm, e o tamanho em px dá ``px_por_recorte`` por recorte.
        passo = tile_mm * (1 - overlap)
        largura_mm = tile_mm + (colunas - 1) * passo
        altura_q_mm = tile_mm + (linhas - 1) * passo
        px_mm = px_por_recorte / tile_mm
        self.size = (round(largura_mm * px_mm), round(altura_q_mm * px_mm))
        self.hfov_deg = math.degrees(2 * math.atan(largura_mm / 2 / distancia_mm))
        meia = math.tan(math.radians(self.hfov_deg) / 2) * self.size[1] / self.size[0]
        vfov = math.degrees(2 * math.atan(meia))
        self.recortes = plan_tiles(self.size, distancia_mm, altura_mm,
                                   crop=crop if crop in ZONES else None, tile_mm=tile_mm,
                                   overlap=overlap, hfov_deg=self.hfov_deg, vfov_deg=vfov)
        self._index = 0
        self._cache: dict[Path, Image.Image] = {}

    def _foto(self, classe: str, lado_px: tuple[int, int]) -> Image.Image:
        lista = self.fotos.get(classe) or self.fotos["Healthy"]
        caminho = lista[int(self.rng.integers(0, len(lista)))]
        if caminho not in self._cache:
            self._cache[caminho] = Image.open(caminho).convert("RGB")
        return self._cache[caminho].resize(lado_px)

    def capture(self) -> Frame:
        indice = self._index
        self._index += 1
        if not self.recortes:
            raise HardwareUnavailableError(f"{self.camera_id}: quadro sem recorte (geometria)")
        fix = self.gps.read()
        planta, _d = self.campo.mais_proxima(fix.lat, fix.lon)
        doenca = CLASSE_DA_CONDICAO.get(planta.condicao, "Healthy")
        quadro = Image.new("RGB", self.size, (30, 60, 25))
        # Em ordem: o recorte seguinte cobre a borda do anterior (sobreposição), como
        # folhas vizinhas entram na borda de cada recorte no campo.
        for t in self.recortes:
            x0, y0, x1, y1 = t.box
            classe = (doenca if doenca != "Healthy" and self.rng.random() < self.incidencia
                      else "Healthy")
            quadro.paste(self._foto(classe, (x1 - x0, y1 - y0)), (x0, y0))
        saida = io.BytesIO()
        quadro.save(saida, "JPEG", quality=92)
        return Frame(self.camera_id, indice, self.clock.now(), saida.getvalue(), "image/jpeg",
                     self.size[0], self.size[1], source=f"simulador-folhas:{planta.id}",
                     distance_mm=self.distancia_mm, camera_height_mm=self.altura_mm,
                     side=self.lado, hfov_deg=self.hfov_deg)

    def describe(self) -> DeviceInfo:
        return DeviceInfo("camera", "câmera de folhas reais (SIMULAÇÃO)", True,
                          {"camera_id": self.camera_id, "recortes": len(self.recortes),
                           "px": list(self.size), "hfov_deg": round(self.hfov_deg, 1)})
