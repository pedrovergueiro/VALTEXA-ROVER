"""Decodificação e pré-processamento de imagens para inferência.

Este módulo é a **fonte única de verdade** do pré-processamento: o pipeline de
treino (:mod:`training.dataset`) importa daqui as mesmas operações. Qualquer
divergência entre treino e inferência não gera erro — ela só derruba a acurácia
em produção, silenciosamente —, por isso as duas pontas compartilham este código.

O pipeline reproduz **exatamente** o do treino (``training.dataset._decode``):

1. decodifica os pixels **como estão no arquivo**, sem aplicar a rotação do EXIF
   — o ``tf.io.decode_image`` do treino não a aplica;
2. redimensiona para ``img_size`` x ``img_size`` com o bilinear do
   ``tf.image.resize`` (centros de pixel, sem antialias) e **sem preservar o
   aspect ratio** — reimplementado em :func:`resize_like_tensorflow`;
3. escala os pixels de ``[0, 255]`` para ``[-1, 1]``, igual a
   ``keras.applications.mobilenet_v2.preprocess_input``.

## Por que exatamente, e não "parecido"

Até 26/09/2026 a API girava a foto pelo EXIF e redimensionava com o bilinear do
Pillow — "a mesma semântica, outra implementação", registrada no README como
folga conhecida. Medida nas mesmas imagens de teste, pelo mesmo modelo, com TTA:

=================  ==============  ==========
Modelo             API de antes    Treino
=================  ==============  ==========
coffee-foliar      92,29%          94,99%
leaf-foliar-b      95,35%          96,01%
leaf-foliar        95,49%          95,55%
=================  ==============  ==========

A folga custava 2,7 pontos no café. Duas causas: o Pillow amplia as fotos de
128 px do café de outro jeito, e um terço das fotos de milho traz rotação no
EXIF — girada, a foto é espremida para o quadrado na outra direção.
"""

from __future__ import annotations

import io
from collections.abc import Iterable

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_IMAGE_PIXELS = 80_000_000
"""Teto de pixels por imagem, como defesa contra *decompression bombs*.

Cabe folgadamente uma foto de drone de 80 MP. O valor também é aplicado ao
limite global do Pillow como segunda camada de proteção.
"""

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


class InvalidImageError(ValueError):
    """A imagem enviada não pôde ser decodificada ou é grande demais."""


def scale_to_unit_range(pixels):
    """Escala pixels de ``[0, 255]`` para ``[-1, 1]``.

    Equivalente a ``keras.applications.mobilenet_v2.preprocess_input``. É escrito
    com aritmética simples de propósito: funciona tanto em ``numpy.ndarray``
    (inferência) quanto em tensores do TensorFlow (pipeline ``tf.data`` do
    treino), sem obrigar a API a importar o Keras só para normalizar.
    """
    return pixels / 127.5 - 1.0


def load_image_from_bytes(
    data: bytes, filename: str = "upload", apply_exif: bool = False
) -> Image.Image:
    """Decodifica os bytes de um upload em uma imagem RGB.

    Por padrão **não** aplica a rotação do EXIF, porque o treino não aplica: o
    modelo aprendeu com os pixels como estão no arquivo. ``apply_exif=True``
    serve a quem vai mostrar a foto para uma pessoa.

    Raises:
        InvalidImageError: bytes vazios, formato não reconhecido ou imagem
            acima de :data:`MAX_IMAGE_PIXELS`.
    """
    if not data:
        raise InvalidImageError(f"{filename}: arquivo vazio.")

    try:
        with Image.open(io.BytesIO(data)) as image:
            # image.size vem do cabeçalho, antes de decodificar os pixels — é o
            # que permite recusar uma bomba sem alocar memória por ela.
            _reject_oversized(image, filename)
            oriented = ImageOps.exif_transpose(image) if apply_exif else image
            return oriented.convert("RGB")
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as exc:
        raise InvalidImageError(f"{filename}: não foi possível ler a imagem ({exc}).") from exc


def resize_like_tensorflow(pixels: np.ndarray, size: int) -> np.ndarray:
    """``tf.image.resize(..., method="bilinear")`` em NumPy, sem o TensorFlow.

    Centros de pixel (``half_pixel_centers``), sem antialias, borda repetida —
    a conta que o treino fez em cada imagem. Devolve ``float32`` em ``[0, 255]``.
    A igualdade com o TensorFlow é conferida em ``tests/test_preprocess.py``
    quando ele está instalado.
    """
    pixels = np.asarray(pixels)
    rows = _bilinear_axis(size, pixels.shape[0])
    cols = _bilinear_axis(size, pixels.shape[1])

    top = pixels[rows[0]].astype(np.float32)
    bottom = pixels[rows[1]].astype(np.float32)
    # Mesma ordem de operações do kernel do TensorFlow: horizontal, depois vertical.
    top_row = top[:, cols[0]] + (top[:, cols[1]] - top[:, cols[0]]) * cols[2][None, :, None]
    bottom_row = (
        bottom[:, cols[0]] + (bottom[:, cols[1]] - bottom[:, cols[0]]) * cols[2][None, :, None]
    )
    return top_row + (bottom_row - top_row) * rows[2][:, None, None]


def _bilinear_axis(out_size: int, in_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vizinhos e peso de cada saída ao longo de um eixo, como o TensorFlow calcula."""
    position = (np.arange(out_size, dtype=np.float32) + 0.5) * (in_size / out_size) - 0.5
    floor = np.floor(position)
    lower = np.maximum(floor, 0).astype(np.intp)
    upper = np.minimum(np.ceil(position), in_size - 1).astype(np.intp)
    return lower, upper, (position - floor).astype(np.float32)


def image_to_array(image: Image.Image, img_size: int, resampling: str = "tensorflow") -> np.ndarray:
    """Converte uma imagem RGB no tensor ``(img_size, img_size, 3)`` do modelo.

    ``resampling="pillow"`` é para o modelo de severidade, que foi treinado com o
    redimensionamento do Pillow (``training.train_severity``).
    """
    if resampling == "pillow":
        resized = image.resize((img_size, img_size), Image.Resampling.BILINEAR)
        pixels = np.asarray(resized, dtype=np.float32)
    else:
        pixels = resize_like_tensorflow(np.asarray(image), img_size)
    return scale_to_unit_range(pixels)


def preprocess_batch(
    images: Iterable[Image.Image], img_size: int, resampling: str = "tensorflow"
) -> np.ndarray:
    """Empilha várias imagens em um único lote ``(N, img_size, img_size, 3)``.

    Uma passada só pelo modelo é bem mais barata que N passadas de tamanho 1.
    """
    arrays = [image_to_array(image, img_size, resampling) for image in images]
    if not arrays:
        raise InvalidImageError("Nenhuma imagem para processar.")
    return np.stack(arrays, axis=0)


def preprocess_image(image: Image.Image, img_size: int) -> np.ndarray:
    """Converte uma imagem no lote unitário ``(1, img_size, img_size, 3)``."""
    return preprocess_batch([image], img_size)


def _reject_oversized(image: Image.Image, filename: str) -> None:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise InvalidImageError(f"{filename}: imagem com dimensões inválidas ({width}x{height}).")
    if width * height > MAX_IMAGE_PIXELS:
        raise InvalidImageError(
            f"{filename}: imagem de {width}x{height} excede o limite de "
            f"{MAX_IMAGE_PIXELS:,} pixels."
        )
