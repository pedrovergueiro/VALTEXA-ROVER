"""Profundidade contra a trena: o operador mede a distância real e escolhe a região.

O operador põe um alvo plano (papelão, a régua) a uma distância medida com trena — o
guia pede 0,3 / 0,5 / 0,8 m, um ensaio por distância — e informa a distância e a região
do quadro onde o alvo está (``roi``: ``[x, y, largura, altura]`` em fração do quadro).
O ensaio lê ``amostras`` mapas de profundidade da câmera, tira a mediana dos pixels
válidos (> 0) da região em cada um e compara a mediana dessas medianas com a trena.

Precisa de câmera com ``capture_depth()`` (OAK-D: o mapa da ``StereoDepth`` alinhado à
foto, em mm).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from valtexa_rover.ensaios.base import (
    Contexto,
    Ensaio,
    EnsaioIndisponivelError,
    Param,
    ParametroInvalidoError,
    criterio,
)
from valtexa_rover.ensaios.estatistica import desvio, mediana


def recorte(depth: np.ndarray, roi: list[float]) -> np.ndarray:
    x, y, w, h = roi
    altura, largura = depth.shape[:2]
    x0, y0 = round(x * largura), round(y * altura)
    x1, y1 = round((x + w) * largura), round((y + h) * altura)
    return depth[max(0, y0):min(altura, y1), max(0, x0):min(largura, x1)]


class Profundidade(Ensaio):
    nome = "profundidade"
    fase = 0
    titulo = "Profundidade contra a trena"
    parametros = (
        Param("camera_id", str, None, descricao="câmera", obrigatorio=True),
        Param("distancia_real_mm", float, None, 100.0, 10000.0,
              "distância medida com trena (mm)", obrigatorio=True),
        Param("roi", list, [0.4, 0.4, 0.2, 0.2], descricao="[x, y, largura, altura] em fração"),
        Param("amostras", int, 10, 1, 500, "mapas de profundidade lidos"),
        Param("intervalo_s", float, 0.2, 0.0, 10.0, "espera entre leituras (s)"),
        Param("tolerancia_mm", float, 20.0, 0.0, 1000.0, "erro máximo aceito (mm)"),
        Param("preenchimento_min", float, 0.5, 0.0, 1.0,
              "fração mínima de pixels com medida na região"),
    )
    hardware_real = ("simulado/não testado em hardware: o alinhamento da StereoDepth à foto "
                     "de 12 MP da OAK-D Pro W ainda não foi conferido em bancada.")

    def executar(self, ctx: Contexto, p: dict[str, Any]) -> dict[str, Any]:
        roi = p["roi"]
        if (len(roi) != 4 or min(roi) < 0 or roi[2] <= 0 or roi[3] <= 0
                or roi[0] + roi[2] > 1 or roi[1] + roi[3] > 1):
            raise ParametroInvalidoError("roi: [x, y, largura, altura] dentro de 0..1")
        camera = next((c for c in ctx.hardware.cameras if c.camera_id == p["camera_id"]), None)
        if camera is None:
            raise EnsaioIndisponivelError(f"câmera {p['camera_id']!r} não existe neste rover")
        if not hasattr(camera, "capture_depth"):
            raise EnsaioIndisponivelError(f"câmera {p['camera_id']!r} não mede profundidade")
        medianas: list[float] = []
        preenchimentos: list[float] = []
        falhas = 0
        for i in range(p["amostras"]):
            try:
                regiao = recorte(np.asarray(camera.capture_depth()), roi)
            except Exception:  # leitura que falha conta, não para o ensaio
                falhas += 1
                regiao = None
            if regiao is not None and regiao.size:
                validos = regiao[regiao > 0]
                preenchimentos.append(float(validos.size) / float(regiao.size))
                if validos.size:
                    medianas.append(float(np.median(validos)))
            ctx.avisa((i + 1) / p["amostras"], f"leitura {i + 1}/{p['amostras']}",
                      {"mediana_mm": medianas[-1] if medianas else None})
            ctx.espera(p["intervalo_s"])
        real = p["distancia_real_mm"]
        medida = mediana(medianas) if medianas else None
        erro = None if medida is None else medida - real
        preench = sum(preenchimentos) / len(preenchimentos) if preenchimentos else 0.0
        criterios = [
            criterio("erro_abs_mm", None if erro is None else abs(erro), p["tolerancia_mm"],
                     "<=", "mm"),
            criterio("preenchimento", preench, p["preenchimento_min"], ">="),
        ]
        medidas = {
            "distancia_real_mm": real,
            "mediana_mm": None if medida is None else round(medida, 1),
            "erro_mm": None if erro is None else round(erro, 1),
            "erro_pct": None if erro is None else round(100 * erro / real, 2),
            "desvio_mm": round(desvio(medianas), 2), "preenchimento": round(preench, 3),
            "leituras": len(medianas), "falhas": falhas,
        }
        return {"criterios": criterios, "medidas": medidas,
                "amostras": {"distancia_real_mm": real,
                             "medianas_mm": [round(m, 2) for m in medianas],
                             "preenchimentos": [round(x, 4) for x in preenchimentos]}}
