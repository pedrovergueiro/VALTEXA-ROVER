"""Vídeo ao vivo para a estação de pilotagem (centro de comando da plataforma).

Liga só quando alguém assiste: a plataforma põe o pedido (câmeras, fps, lado) no
cabeçalho ``X-Valtexa-Video`` da resposta de ``GET /v1/device/commands`` — a consulta que
o Runtime já faz — e na resposta de cada quadro. O pedido vence sozinho (``ate``):
fechou a tela, o robô para de mandar sem precisar de comando.

**De onde vem a imagem, sem atrapalhar a IA:**

* com missão em curso, o laço do Runtime já captura todas as câmeras: o vídeo usa o
  último quadro de cada uma (``RoverCore.ultimos_quadros``) e **nunca** chama a câmera;
* fora de missão, captura direto — pela saída leve da câmera quando ela tem
  (``capture_preview``: a OAK-D tem uma saída de vídeo separada da foto parada de 12 MP),
  senão pela foto reduzida aqui;
* com ensaio em curso, não captura (o ensaio mede a câmera; o vídeo não pode disputar).

**Qualidade adaptativa:** cada quadro é um pedido síncrono; se ele demora mais que o
intervalo pedido, o próximo sai menor (lado e qualidade JPEG descem um degrau); se sobra
folga, sobe. No rádio lento o operador vê menos detalhe, não vídeo atrasado. Pela base
da fazenda, cada câmera é um slot "só o último vale" (``enlace/rpc.py``): quadro velho é
descartado no caminho.
"""

from __future__ import annotations

import io
import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from valtexa_rover._compat import UTC
from valtexa_rover.hal.interfaces import HardwareUnavailableError, NetworkUnavailableError

log = logging.getLogger("valtexa_rover.video")

CABECALHO = "X-Valtexa-Video"
DEGRAUS: tuple[tuple[int, int], ...] = ((240, 50), (320, 55), (480, 62), (640, 70),
                                        (800, 75), (960, 80), (1280, 85))
"""(lado maior em px, qualidade JPEG), do mais leve ao melhor."""


@dataclass(frozen=True)
class PedidoDeVideo:
    cameras: tuple[str, ...]
    fps: float
    lado: int
    ate: datetime

    @classmethod
    def de_json(cls, bruto: str | dict | None) -> PedidoDeVideo | None:
        if not bruto:
            return None
        try:
            d = json.loads(bruto) if isinstance(bruto, str) else bruto
            cameras = tuple(str(c) for c in d["cameras"])[:4]
            ate = datetime.fromisoformat(d["ate"])
            if ate.tzinfo is None:
                ate = ate.replace(tzinfo=UTC)
            fps = min(10.0, max(0.5, float(d["fps"])))
            lado = min(1280, max(240, int(d["lado"])))
        except (ValueError, KeyError, TypeError):
            return None
        return cls(cameras, fps, lado, ate) if cameras else None


def reduz(jpeg: bytes, lado: int, qualidade: int) -> tuple[bytes, int, int]:
    """JPEG com o lado maior em ``lado`` px (nunca aumenta)."""
    from PIL import Image

    with Image.open(io.BytesIO(jpeg)) as im:
        im = im.convert("RGB")
        im.thumbnail((lado, lado))
        saida = io.BytesIO()
        im.save(saida, "JPEG", quality=qualidade)
        return saida.getvalue(), im.width, im.height


class TransmissorDeVideo:
    def __init__(self, hardware, client, clock, *,
                 missao_em_curso: Callable[[], bool] = lambda: False,
                 ocupado: Callable[[], bool] = lambda: False,
                 quadro_recente: Callable[[str], object | None] = lambda _c: None,
                 dormir: Callable[[float], None] | None = None) -> None:
        self.hardware = hardware
        self.client = client
        self.clock = clock
        self.missao_em_curso = missao_em_curso
        self.ocupado = ocupado
        self.quadro_recente = quadro_recente
        self._pedido: PedidoDeVideo | None = None
        self._trava = threading.Lock()
        self._parar = threading.Event()
        self._dormir = dormir or self._parar.wait
        self._thread: threading.Thread | None = None
        self.degrau = 3  # 640 px, qualidade 70
        self.seq = 0
        self.enviados = 0
        self.descartados = 0
        self.ultimo_erro: str | None = None
        self.bytes_enviados = 0
        self.falhas = 0

    # -- pedido -----------------------------------------------------------------

    def atualiza(self, pedido: PedidoDeVideo | str | dict | None) -> None:
        """Chamado com o cabeçalho de cada consulta de comandos (``None``: ninguém assiste)."""
        if not isinstance(pedido, PedidoDeVideo):
            pedido = PedidoDeVideo.de_json(pedido)
        with self._trava:
            self._pedido = pedido

    def pedido(self) -> PedidoDeVideo | None:
        with self._trava:
            p = self._pedido
        if p is not None and p.ate <= self.clock.now():
            return None
        return p

    # -- laço ---------------------------------------------------------------------

    def iniciar(self) -> TransmissorDeVideo:
        self._thread = threading.Thread(target=self._laco, name="video-ao-vivo", daemon=True)
        self._thread.start()
        return self

    def parar(self) -> None:
        self._parar.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _laco(self) -> None:
        """Nunca morre: um erro inesperado vira ``ultimo_erro`` e o laço segue (o vídeo é
        o olho do operador; parar calado seria pior que um quadro perdido)."""
        while not self._parar.is_set():
            try:
                mandou = self.passo()
            except Exception as exc:
                self.falhas += 1
                self.ultimo_erro = f"{type(exc).__name__}: {exc}"
                log.warning("vídeo ao vivo: %s", self.ultimo_erro)
                mandou = False
            if not mandou:
                self._dormir(0.2)

    def passo(self) -> bool:
        """Manda um quadro de cada câmera pedida. ``False`` se não havia o que mandar."""
        p = self.pedido()
        if p is None:
            return False
        periodo = 1.0 / p.fps
        inicio = time.monotonic()
        mandou = False
        for camera_id in p.cameras:
            mandou = self._um(camera_id, p, periodo / max(1, len(p.cameras))) or mandou
        resto = periodo - (time.monotonic() - inicio)
        if resto > 0:
            self._dormir(resto)
        return mandou

    def _limite(self, p: PedidoDeVideo) -> int:
        return max(i for i, (lado, _q) in enumerate(DEGRAUS) if lado <= p.lado or i == 0)

    def _imagem(self, camera_id: str, lado: int, qualidade: int):
        """``(jpeg, w, h, capturado_em)`` ou ``None`` (sem quadro agora)."""
        if self.missao_em_curso():
            f = self.quadro_recente(camera_id)
            if f is None:
                return None
            jpeg, w, h = reduz(f.data, lado, qualidade)
            return jpeg, w, h, f.captured_at
        if self.ocupado():
            return None
        camera = next((c for c in self.hardware.cameras if c.camera_id == camera_id), None)
        if camera is None:
            return None
        previa = getattr(camera, "capture_preview", None)
        if previa is not None:
            jpeg, w, h, quando = previa(lado, qualidade)
            return jpeg, w, h, quando
        f = camera.capture()
        jpeg, w, h = reduz(f.data, lado, qualidade)
        return jpeg, w, h, f.captured_at

    def _um(self, camera_id: str, p: PedidoDeVideo, fatia_s: float) -> bool:
        self.degrau = min(self.degrau, self._limite(p))
        lado, qualidade = DEGRAUS[self.degrau]
        try:
            imagem = self._imagem(camera_id, lado, qualidade)
        except HardwareUnavailableError as exc:
            self.ultimo_erro = f"{camera_id}: {exc}"
            return False
        if imagem is None:
            return False
        jpeg, _w, _h, quando = imagem
        self.seq += 1
        t0 = time.monotonic()
        try:
            r = self.client.request(
                "POST", f"/v1/device/video/{camera_id}", raw=jpeg, content_type="image/jpeg",
                extra_headers={"X-Capturado-Em": quando.isoformat(), "X-Seq": str(self.seq)})
        except NetworkUnavailableError as exc:
            self.descartados += 1
            self.ultimo_erro = str(exc)
            self.degrau = max(0, self.degrau - 1)
            return False
        gasto = time.monotonic() - t0
        if r.status in (404, 409):  # ninguém assiste mais
            self.atualiza(None)
            return False
        if r.status != 200:
            self.descartados += 1
            self.ultimo_erro = f"HTTP {r.status}"
            return False
        self.enviados += 1
        self.bytes_enviados += len(jpeg)
        self.atualiza(PedidoDeVideo.de_json(r.json()) or p)
        # adapta: estourou a fatia → mais leve; sobrou muito → melhor
        if gasto > fatia_s * 0.9:
            self.degrau = max(0, self.degrau - 1)
        elif gasto < fatia_s * 0.35:
            self.degrau = min(self._limite(p), self.degrau + 1)
        return True

    def estado(self) -> dict:
        p = self.pedido()
        return {"ativo": p is not None, "cameras": list(p.cameras) if p else [],
                "degrau": DEGRAUS[self.degrau], "enviados": self.enviados,
                "descartados": self.descartados, "bytes": self.bytes_enviados,
                "ultimo_erro": self.ultimo_erro}
