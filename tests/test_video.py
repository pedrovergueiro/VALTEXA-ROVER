"""Vídeo ao vivo do Rover: pedido, origem da imagem, qualidade adaptativa e o enlace."""

from __future__ import annotations

import io
import json
import time
from datetime import datetime, timedelta

import numpy as np
from PIL import Image

from valtexa_rover._compat import UTC
from valtexa_rover.clock import ManualClock
from valtexa_rover.enlace.rpc import P3, PRAZO_VIDEO, classifica
from valtexa_rover.hal.interfaces import Frame, HardwareUnavailableError, HttpResponse
from valtexa_rover.runtime.video import DEGRAUS, PedidoDeVideo, TransmissorDeVideo

AGORA = datetime(2026, 10, 10, 12, tzinfo=UTC)


def _jpeg(lado=(1600, 1200), cor=(30, 140, 40)) -> bytes:
    b = io.BytesIO()
    Image.new("RGB", lado, cor).save(b, "JPEG")
    return b.getvalue()


class Camera:
    def __init__(self, camera_id="cam-0", falha=False):
        self.camera_id = camera_id
        self.capturas = 0
        self.falha = falha

    def capture(self) -> Frame:
        if self.falha:
            raise HardwareUnavailableError("câmera fora")
        self.capturas += 1
        return Frame(self.camera_id, self.capturas, AGORA, _jpeg(), "image/jpeg", 1600, 1200)


class Hw:
    def __init__(self, *cameras):
        self.cameras = list(cameras)


class Cliente:
    def __init__(self, status=200, atraso=0.0, pedido=None):
        self.pedidos = []
        self.status = status
        self.atraso = atraso
        self.pedido = pedido

    def request(self, method, path, *, raw=None, content_type=None, extra_headers=None, **_):
        if self.atraso:
            time.sleep(self.atraso)
        self.pedidos.append((method, path, raw, extra_headers))
        corpo = json.dumps(self.pedido).encode() if self.pedido else b"{}"
        return HttpResponse(self.status, corpo)


def _pedido(cameras=("cam-0",), fps=5.0, lado=640, vence=timedelta(seconds=10)) -> dict:
    return {"cameras": list(cameras), "fps": fps, "lado": lado,
            "ate": (AGORA + vence).isoformat()}


def _tx(hw, cliente, **kw) -> TransmissorDeVideo:
    return TransmissorDeVideo(hw, cliente, ManualClock(AGORA), dormir=lambda _s: None, **kw)


def test_pedido_vence_e_e_validado() -> None:
    assert PedidoDeVideo.de_json(None) is None
    assert PedidoDeVideo.de_json('{"cameras": []}') is None
    p = PedidoDeVideo.de_json(json.dumps(_pedido(fps=99, lado=5000)))
    assert p.fps == 10.0 and p.lado == 1280  # limitado do lado do robô também
    tx = _tx(Hw(Camera()), Cliente())
    tx.atualiza(json.dumps(_pedido(vence=timedelta(seconds=-1))))
    assert tx.pedido() is None and tx.passo() is False  # vencido: não manda nada


def test_fora_de_missao_captura_e_manda_jpeg_reduzido() -> None:
    cam, cli = Camera(), Cliente(pedido=_pedido())
    tx = _tx(Hw(cam), cli)
    tx.atualiza(_pedido())
    assert tx.passo() is True
    metodo, caminho, jpeg, cab = cli.pedidos[0]
    assert (metodo, caminho) == ("POST", "/v1/device/video/cam-0")
    with Image.open(io.BytesIO(jpeg)) as im:
        assert max(im.size) <= 640
    assert cab["X-Seq"] == "1" and cab["X-Capturado-Em"].startswith("2026-10-10")


def test_em_missao_usa_o_quadro_do_runtime_e_nunca_chama_a_camera() -> None:
    cam = Camera(falha=True)  # se fosse chamada, falharia
    recente = Frame("cam-0", 7, AGORA, _jpeg(), "image/jpeg", 1600, 1200)
    cli = Cliente(pedido=_pedido())
    tx = _tx(Hw(cam), cli, missao_em_curso=lambda: True, quadro_recente={"cam-0": recente}.get)
    tx.atualiza(_pedido())
    assert tx.passo() is True and len(cli.pedidos) == 1


def test_ensaio_em_curso_nao_disputa_a_camera() -> None:
    cam, cli = Camera(), Cliente(pedido=_pedido())
    tx = _tx(Hw(cam), cli, ocupado=lambda: True)
    tx.atualiza(_pedido())
    assert tx.passo() is False and cam.capturas == 0 and cli.pedidos == []


def test_qualidade_adapta_ao_enlace() -> None:
    lento = Cliente(atraso=0.12, pedido=_pedido(fps=10))  # 120 ms por quadro a 10 fps
    tx = _tx(Hw(Camera()), lento)
    tx.atualiza(_pedido(fps=10))
    inicio = tx.degrau
    for _ in range(3):
        tx.passo()
    assert tx.degrau < inicio  # mais leve: menos detalhe, não vídeo atrasado
    rapido = Cliente(pedido=_pedido(fps=1, lado=960))
    tx2 = _tx(Hw(Camera()), rapido)
    tx2.atualiza(_pedido(fps=1, lado=960))
    for _ in range(5):
        tx2.passo()
    assert DEGRAUS[tx2.degrau][0] == 960  # sobe até o pedido, nunca acima


def test_ninguem_assiste_409_para() -> None:
    cli = Cliente(status=409)
    tx = _tx(Hw(Camera()), cli)
    tx.atualiza(_pedido())
    tx.passo()
    assert tx.pedido() is None


def test_executor_de_comandos_repassa_o_cabecalho(tmp_path) -> None:
    from valtexa_rover.runtime.comandos import CommandExecutor

    class Core:
        pass

    core = Core()
    core.hardware = Hw(Camera())
    core.store = type("S", (), {"root": tmp_path})()
    core.clock = ManualClock(AGORA)
    core.identity = type("I", (), {"platform_public_key": "00" * 32, "device_id": "d"})()
    resposta = HttpResponse(200, b"[]", {"X-Valtexa-Video": json.dumps(_pedido())})
    core.client = type("C", (), {"request": lambda *_a, **_k: resposta})()
    ex = CommandExecutor.__new__(CommandExecutor)
    ex.core, ex.hardware, ex._laudos_pendentes = core, core.hardware, []
    ex._lock_laudos = __import__("threading").Lock()
    ex.video = _tx(Hw(Camera()), Cliente())
    ex.poll()
    assert ex.video.pedido() is not None and ex.video.pedido().cameras == ("cam-0",)


def test_enlace_video_e_slot_so_o_ultimo_abaixo_de_comando() -> None:
    prio, slot = classifica("POST", "/v1/device/video/cam-2", b"")
    assert prio == P3 and slot == "video:cam-2"
    assert classifica("GET", "/v1/device/commands", b"")[0] < P3
    assert PRAZO_VIDEO <= 3.0


def test_previa_da_oak_d_com_sdk_falso() -> None:
    from valtexa_rover.hal.rover_v3 import OakDCamera

    class No:
        def __getattr__(self, _):
            return self

        def __call__(self, *a, **k):
            return self

        def link(self, *_):
            pass

    class Fila:
        def __init__(self, item=None):
            self.item = item

        def get(self):
            return self.item

        def tryGet(self):  # noqa: N802 (API do depthai)
            return self.item

        def send(self, _x):
            pass

    class Pacote:
        def __init__(self, quadro):
            self.quadro = quadro

        def getFrame(self):  # noqa: N802
            return self.quadro

    rgb = np.zeros((720, 960, 3), np.uint8)
    rgb[..., 1] = 180

    class Dispositivo:
        def __init__(self, *_):
            self.filas = {"preview": Fila(Pacote(rgb)), "control": Fila(),
                          "still": Fila(), "depth": Fila()}

        def getOutputQueue(self, nome, **_):  # noqa: N802
            return self.filas[nome]

        def getInputQueue(self, nome):  # noqa: N802
            return self.filas[nome]

    class Dai(No):
        Device = Dispositivo

    cam = OakDCamera("esq-1000", "192.168.10.12", 1000.0, dai=Dai(), undistort=False)
    jpeg, w, h, _quando = cam.capture_preview(480, 60)
    assert (w, h) == (480, 360)
    with Image.open(io.BytesIO(jpeg)) as im:
        assert im.getpixel((10, 10))[1] > 150


def test_laco_nao_morre_com_erro_inesperado() -> None:
    chamadas = []

    def explode():
        chamadas.append(1)
        raise TypeError("defeito de quem montou o transmissor")

    tx = TransmissorDeVideo(Hw(Camera()), Cliente(pedido=_pedido()), ManualClock(AGORA),
                            ocupado=explode, dormir=lambda _s: time.sleep(0.01))
    tx.atualiza(_pedido())
    tx.iniciar()
    time.sleep(0.3)
    tx.parar()
    assert len(chamadas) > 1  # seguiu tentando depois do primeiro erro
    assert tx.falhas >= 1 and "TypeError" in tx.ultimo_erro
