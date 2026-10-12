"""Diagnóstico peça a peça: o mesmo laudo no terminal (``valtexa-rover diagnostico``) e no
centro de comando da plataforma (comando remoto ``diagnostico``).

Uma leitura de cada peça; ``ok: false`` com o erro em vez de parar na primeira. Peças:
``camera:<id>`` (ou ``cameras``, todas), ``gnss``, ``imu``, ``acelerador``, ``sistema``,
``disco``, ``navegacao``, ``lente`` (a correção de distorção de cada câmera).
"""

from __future__ import annotations

import base64
import dataclasses
import io
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from valtexa_rover.hal.interfaces import Hardware

TODAS = ("cameras", "gnss", "imu", "acelerador", "sistema", "disco", "navegacao", "lente")


def _quer(pecas: list[str] | None, nome: str) -> bool:
    if pecas is None:
        return True
    return nome in pecas or (nome.startswith("camera:") and "cameras" in pecas)


def thumbnail(jpeg: bytes, lado: int = 640, quality: int = 80) -> dict[str, Any]:
    """A foto reduzida em base64, para o painel (a original fica no robô)."""
    from PIL import Image

    with Image.open(io.BytesIO(jpeg)) as im:
        im = im.convert("RGB")
        im.thumbnail((lado, lado))
        saida = io.BytesIO()
        im.save(saida, "JPEG", quality=quality)
        return {"jpeg_base64": base64.b64encode(saida.getvalue()).decode("ascii"),
                "thumb_px": [im.width, im.height]}


def run(hardware: Hardware, pecas: list[str] | None = None, *, salvar: Path | None = None,
        espera_gnss: float = 5.0, fotos: bool = False) -> dict[str, Any]:
    """O laudo: ``{"ok", "hardware", "pecas": {nome: {"ok", ...}}}``.

    ``fotos``: cada câmera devolve também a foto reduzida (o centro de comando mostra).
    """
    laudo: dict[str, dict] = {}

    def medir(nome: str, leitura: Callable[[], dict]) -> None:
        if not _quer(pecas, nome):
            return
        try:
            laudo[nome] = {"ok": True, **leitura()}
        except Exception as exc:  # peça que não responde: o laudo diz qual
            laudo[nome] = {"ok": False, "erro": f"{type(exc).__name__}: {exc}"}

    for camera in hardware.cameras:
        def quadro(camera=camera) -> dict:
            q = camera.capture()
            if salvar is not None:
                salvar.mkdir(parents=True, exist_ok=True)
                (salvar / f"{q.camera_id}.jpg").write_bytes(q.data)
            saida = {"px": [q.width, q.height], "distancia_mm": q.distance_mm,
                     "altura_mm": q.camera_height_mm, "lado": q.side, "hfov_deg": q.hfov_deg,
                     "bytes": len(q.data)}
            if fotos:
                saida.update(thumbnail(q.data))
            return saida
        medir(f"camera:{camera.camera_id}", quadro)

    if _quer(pecas, "lente"):
        def lente(camera) -> str | None:
            try:
                return (camera.describe().detail or {}).get("lente")
            except Exception:  # câmera sem describe (simulada, quebrada)
                return None

        laudo["lente"] = {"ok": all(getattr(c, "undistorter", None) is not None
                                    for c in hardware.cameras if hasattr(c, "undistorter")),
                          "cameras": {c.camera_id: lente(c) for c in hardware.cameras}}

    def gnss() -> dict:
        limite = time.monotonic() + espera_gnss
        while True:  # o adapter lê numa thread: o primeiro fix pode levar uns segundos
            try:
                fix = hardware.gps.read()
                break
            except Exception:
                if time.monotonic() >= limite:
                    raise
                time.sleep(0.2)
        return {k: (v.isoformat() if hasattr(v, "isoformat") else v)
                for k, v in dataclasses.asdict(fix).items()}

    medir("gnss", gnss)
    if hardware.imu is not None:
        def imu() -> dict:
            time.sleep(0.5)  # a thread do CAN recebe as primeiras mensagens
            leitura = hardware.imu.read()
            return {"roll_pitch_yaw_deg": [leitura.roll_deg, leitura.pitch_deg,
                                           leitura.yaw_deg],
                    "accel_ms2": list(leitura.accel_ms2)}
        medir("imu", imu)
    medir("acelerador", lambda: {"providers": hardware.accelerator.providers()})
    if hardware.system is not None:
        medir("sistema", lambda: dataclasses.asdict(hardware.system.read()))
    medir("disco", lambda: dict(zip(("total_bytes", "livre_bytes"), hardware.storage.usage(),
                                    strict=True)))
    if hardware.controller is not None:
        medir("navegacao", hardware.controller.state)
    return {"ok": all(p["ok"] for p in laudo.values()), "hardware": hardware.hardware_version,
            "pecas": laudo}
