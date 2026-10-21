"""Percepção ativa: na dúvida, o Rover olha de novo — e só o que continua incerto sobe inteiro.

Cenas da :class:`~valtexa_rover.hal.simulated.SimulatedRgbdCamera` com o modelo minúsculo do
``conftest`` (cor → classe) e o perfil por contagem de ``test_edge_perfil`` (evento ≥ 0,80,
quase-evento ≥ 0,50). A cor ``AMBIGUA`` dá ~0,65 de ferrugem num recorte: quase-evento.

Os três casos que justificam o módulo:

* **reflexo** — uma mancha que só existe na primeira foto: a vista extra não a vê, a média
  cai abaixo do quase-evento e o quadro sai sadio, resolvido;
* **lesão mal exposta** — com a exposição medida na região, a câmera mostra a cor de
  verdade e a média passa do evento: doença confirmada;
* **dúvida que fica** — as vistas concordam na ambiguidade: o quadro sobe em ORIGINAL com as
  vistas, marcado para a validação do agrônomo.
"""

from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from conftest import build_tiny_onnx
from PIL import Image
from test_edge_perfil import PERFIL

from valtexa_rover._compat import UTC
from valtexa_rover.clock import ManualClock
from valtexa_rover.config import (
    ActivePerceptionConfig,
    Category,
    ClassificationConfig,
    ConfigError,
    RuntimeConfig,
    SyncConfig,
)
from valtexa_rover.edge_ai.engine import EdgeAIEngine, EdgeAIError, ModelPackage
from valtexa_rover.edge_ai.perfil import EdgeProfile, ProfileError
from valtexa_rover.hal.interfaces import HardwareUnavailableError
from valtexa_rover.hal.local import OnnxCpuAccelerator
from valtexa_rover.hal.simulated import SimulatedRgbdCamera
from valtexa_rover.runtime.percepcao import (
    ActivePerception,
    assess,
    fuse,
    gray_small,
    plausible_shift,
    register,
)

AMBIGUA = (195, 185, 0)
VERMELHO = (255, 0, 0)
MANCHA = {"x": 650, "y": 450, "lado": 300}


@pytest.fixture
def relogio() -> ManualClock:
    return ManualClock(datetime(2026, 10, 15, 12, 0, tzinfo=UTC))


def motor(tmp_path: Path, perfil: dict | None = None) -> EdgeAIEngine:
    caminho = tmp_path / "m.onnx"
    sha = build_tiny_onnx(caminho)
    caminho.with_suffix(".edge.json").write_text(json.dumps(perfil or PERFIL), encoding="utf-8")
    m = EdgeAIEngine(OnnxCpuAccelerator(), ClassificationConfig())
    m.load(ModelPackage("coffee-foliar", "1", sha, caminho))
    return m


def cena(relogio, **mancha) -> SimulatedRgbdCamera:
    return SimulatedRgbdCamera("esq-1000", relogio, vaos=[(300, 360)],
                               manchas=[{**MANCHA, **mancha}])


def primeira(m: EdgeAIEngine, camera: SimulatedRgbdCamera):
    quadro = camera.capture()
    return quadro, m.infer(quadro.data, quadro.distance_mm, quadro.camera_height_mm,
                           depth=quadro.depth)


class TestAlinhamento:
    def test_correlacao_de_fase_acha_o_passo_do_rover(self, relogio) -> None:
        camera = SimulatedRgbdCamera("c", relogio, passo_px=37, vaos=[(300, 360)])
        a, b = camera.capture(), camera.capture()
        (ga, ea), (gb, _) = gray_small(a.data), gray_small(b.data)
        dx, dy, pico = register(ga, gb)
        # ponto_em_a = ponto_em_b + d: a cena andou 37 px para a esquerda entre a e b.
        assert dx * ea == pytest.approx(37.0, abs=4.0) and abs(dy * ea) < 4.0
        assert pico > 0.3

    def test_cenas_sem_relacao_dao_pico_baixo(self, relogio) -> None:
        a = SimulatedRgbdCamera("c", relogio, seed=1).capture()
        b = SimulatedRgbdCamera("c", relogio, seed=99).capture()
        assert register(gray_small(a.data)[0], gray_small(b.data)[0])[2] < 0.06

    def test_deslocamento_fora_do_movimento_do_rover_nao_alinha(self) -> None:
        cfg = ActivePerceptionConfig()
        assert plausible_shift(40.0, 3.0, 1600, 1200, cfg)
        assert not plausible_shift(40.0, 300.0, 1600, 1200, cfg)  # vertical: um reflexo
        assert not plausible_shift(1200.0, 0.0, 1600, 1200, cfg)


class TestDuvida:
    def test_quase_evento_com_regiao_em_volta_da_mancha(self, tmp_path, relogio) -> None:
        m = motor(tmp_path)
        quadro, r = primeira(m, cena(relogio, cor=AMBIGUA))
        assert r.category is Category.NORMAL and r.low_confidence
        duvida = assess(r, quadro, m, ActivePerceptionConfig())
        assert duvida.reasons == ("quase_evento",) and duvida.decision_doubt
        x0, y0, x1, y1 = duvida.region
        assert x0 < 800 / 1600 < x1 and y0 < 600 / 1200 < y1  # o centro da mancha
        assert (x1 - x0) < 0.6  # não é o quadro inteiro

    def test_quadro_claro_nao_tem_duvida(self, tmp_path, relogio) -> None:
        m = motor(tmp_path)
        for cor in (VERMELHO, (0, 255, 0)):
            quadro, r = primeira(m, cena(relogio, cor=cor))
            assert not assess(r, quadro, m, ActivePerceptionConfig()).uncertain

    def test_evidencia_na_fronteira_pela_margem_do_perfil(self, tmp_path, relogio) -> None:
        regra = {"bin_edges": [0.0, 0.5, 0.8, 1.0], "llr_tables": {"Rust": [-0.1, 2.0, 3.0]},
                 "llr_clip": 3.0, "tile_correlation": 1.0}
        m0 = motor(tmp_path / "a", {**PERFIL, "evidence": {**regra, "event_evidence": 0.0,
                                                          "near_miss_evidence": -1.0}})
        quadro, r = primeira(m0, cena(relogio, cor=AMBIGUA))
        escore = r.evidence
        m = motor(tmp_path / "b", {**PERFIL, "evidence": {
            **regra, "event_evidence": escore + 0.01, "near_miss_evidence": escore - 0.02}})
        r = m.infer(quadro.data, quadro.distance_mm, 1000.0, depth=quadro.depth)
        duvida = assess(r, quadro, m, ActivePerceptionConfig())
        assert duvida.reasons == ("evidencia_na_fronteira",)
        assert duvida.margin == pytest.approx(0.03)
        assert duvida.region is not None  # os recortes com LLR a favor da ferrugem
        longe = assess(r, quadro, m, ActivePerceptionConfig(margem_evidencia=0.001))
        assert not longe.uncertain

    def test_sem_perfil_nao_age(self, tmp_path, relogio) -> None:
        caminho = tmp_path / "m.onnx"
        sha = build_tiny_onnx(caminho)
        m = EdgeAIEngine(OnnxCpuAccelerator(), ClassificationConfig())
        m.load(ModelPackage("x", "1", sha, caminho))
        quadro, r = primeira(m, cena(relogio, cor=AMBIGUA))
        assert not assess(r, quadro, m, ActivePerceptionConfig()).uncertain


class TestCiclo:
    def test_reflexo_some_na_vista_extra_e_o_quadro_sai_sadio(self, tmp_path, relogio) -> None:
        m = motor(tmp_path)
        camera = cena(relogio, cor=AMBIGUA, so_na_primeira=True)
        quadro, r = primeira(m, camera)
        ap = ActivePerception(ActivePerceptionConfig(), relogio)
        d = ap.run(m, camera, quadro, r)
        assert d.inference.category is Category.NORMAL and not d.inference.low_confidence
        assert d.block["resolved"] and not d.for_validation and d.extra_frames == []
        vista = d.block["views"][0]
        assert vista["strategy"] == "regiao" and vista["aligned"]
        assert vista["shift_px"][0] == pytest.approx(24.0, abs=4.0)
        assert 0 < vista["tiles"] < len(r.tiles)  # só a região rodou o modelo
        assert d.block["fused_tiles"]["indices"] and set(d.block["fused_tiles"]["views"]) == {2}
        assert camera.regioes == [tuple(d.block["region"])]
        assert d.inference.tiles == r.tiles  # os recortes crus da principal seguem para a nuvem

    def test_lesao_mal_exposta_confirmada_pela_regiao(self, tmp_path, relogio) -> None:
        m = motor(tmp_path)
        camera = cena(relogio, cor=AMBIGUA, cor_regiao=VERMELHO)
        quadro, r = primeira(m, camera)
        d = ActivePerception(ActivePerceptionConfig(), relogio).run(m, camera, quadro, r)
        assert r.category is Category.NORMAL and r.low_confidence
        assert d.inference.category is Category.POSSIVEL_DOENCA and d.inference.label == "Rust"
        assert d.block["resolved"] and "fusão de 2 vistas" in d.inference.reason
        assert d.block["before"]["category"] == "NORMAL"
        assert d.block["after"]["category"] == "POSSIVEL_DOENCA"

    def test_duvida_que_fica_vai_para_a_validacao_com_as_vistas(self, tmp_path,
                                                                 relogio) -> None:
        m = motor(tmp_path)
        camera = cena(relogio, cor=AMBIGUA)
        quadro, r = primeira(m, camera)
        ap = ActivePerception(ActivePerceptionConfig(max_vistas_extras=2), relogio)
        d = ap.run(m, camera, quadro, r)
        assert not d.block["resolved"] and d.for_validation and d.block["for_validation"]
        assert len(d.block["views"]) == 2 and len(d.extra_frames) == 2
        assert all(f.view == "regiao" for f in d.extra_frames)
        assert set(d.block["fused_tiles"]["views"]) == {3}
        assert ap.snapshot()["para_validacao"] == 1 and ap.snapshot()["vistas"] == 2

        nunca = ActivePerception(ActivePerceptionConfig(guardar_vistas="nunca"), relogio)
        quadro, r = primeira(m, camera)
        assert nunca.run(m, camera, quadro, r).extra_frames == []

    def test_orcamento_por_minuto_e_por_tempo(self, tmp_path, relogio) -> None:
        m = motor(tmp_path)
        camera = cena(relogio, cor=AMBIGUA)
        ap = ActivePerception(ActivePerceptionConfig(max_extras_por_minuto=1), relogio)
        quadro, r = primeira(m, camera)
        assert len(ap.run(m, camera, quadro, r).block["views"]) == 1
        quadro, r = primeira(m, camera)
        d = ap.run(m, camera, quadro, r)
        assert d.block["skipped"] == "orcamento" and not d.block["acted"]
        assert d.for_validation  # dúvida sem segunda vista: sobe para o agrônomo
        relogio.advance(61.0)
        quadro, r = primeira(m, camera)
        assert ap.run(m, camera, quadro, r).block["acted"]  # a janela de um minuto passou

        lento = ActivePerception(ActivePerceptionConfig(tempo_max_ms=0.001), relogio)
        quadro, r = primeira(m, cena(relogio, cor=AMBIGUA))
        assert lento.run(m, camera, quadro, r).block["skipped"] == "tempo"

    def test_profundidade_ruim_troca_pela_vista_com_medida(self, tmp_path, relogio) -> None:
        m = motor(tmp_path)
        camera = SimulatedRgbdCamera("c", relogio, cobertura_primeira=0.05, vaos=[(300, 360)])
        quadro, r = primeira(m, camera)
        assert r.rgbd["sufficient"] is False
        d = ActivePerception(ActivePerceptionConfig(), relogio).run(m, camera, quadro, r)
        assert d.block["reasons"] == ["profundidade_insuficiente"]
        assert d.block["promoted_view"] and d.frame is not quadro
        assert d.inference.rgbd["sufficient"] and d.block["resolved"]
        assert not d.for_validation  # problema do sensor, não dúvida do modelo
        assert d.block["views"][0]["region"] is None  # olhou o quadro inteiro

    def test_profundidade_que_nao_volta_nao_sobe_original(self, tmp_path, relogio) -> None:
        m = motor(tmp_path)
        camera = SimulatedRgbdCamera("c", relogio, cobertura=0.05)
        quadro, r = primeira(m, camera)
        d = ActivePerception(ActivePerceptionConfig(), relogio).run(m, camera, quadro, r)
        assert not d.block["resolved"] and not d.for_validation and d.extra_frames == []

    def test_camera_que_falha_na_vista_extra(self, tmp_path, relogio) -> None:
        m = motor(tmp_path)
        camera = cena(relogio, cor=AMBIGUA)
        quadro, r = primeira(m, camera)

        def quebra(*_a, **_k):
            raise HardwareUnavailableError("cabo")

        camera.capture_region = quebra
        ap = ActivePerception(ActivePerceptionConfig(), relogio)
        d = ap.run(m, camera, quadro, r)
        assert d.inference is r and d.block["views"][0]["error"] == "cabo"
        assert ap.snapshot()["falha_captura"] == 1

    def test_pedido_de_pausa_a_navegacao(self, tmp_path, relogio) -> None:
        m = motor(tmp_path)
        camera = cena(relogio, cor=AMBIGUA, so_na_primeira=True)
        pedidos = []

        class Nav:
            def pedir_pausa(self, s, motivo):
                pedidos.append((s, motivo))

        quadro, r = primeira(m, camera)
        d = ActivePerception(ActivePerceptionConfig(pedir_pausa_s=1.5), relogio).run(
            m, camera, quadro, r, Nav())
        assert pedidos == [(1.5, "percepção ativa: esq-1000")]
        assert d.block["pause_requested_s"] == 1.5

    def test_fusao_sem_vista_alinhada_nao_muda_nada(self, tmp_path, relogio) -> None:
        m = motor(tmp_path)
        _quadro, r = primeira(m, cena(relogio, cor=AMBIGUA))
        assert fuse(m, r, []).inference is r


def test_configuracao_da_percepcao_e_validada(tmp_path) -> None:
    caminho = tmp_path / "rover.json"
    for errado, trecho in (({"guardar_vistas": "tudo"}, "guardar_vistas"),
                           ({"tempo_max_ms": 0}, "tempo_max_ms"),
                           ({"max_desvio_vertical": 2}, "max_desvio"),
                           ({"vistas": 3}, "desconhecida")):
        caminho.write_text(json.dumps({"percepcao_ativa": errado}))
        with pytest.raises(ConfigError, match=trecho):
            RuntimeConfig.load(caminho)


class TestCabecaDoPerfil:
    def test_perfil_de_outra_cabeca_e_recusado(self, tmp_path) -> None:
        from test_edge_cabeca import TORRE, modelo_com_cabeca

        onnx = modelo_com_cabeca(tmp_path)
        perfil = json.loads(onnx.with_suffix(".edge.json").read_text(encoding="utf-8"))
        certo = hashlib.sha256(onnx.with_suffix(".head.npz").read_bytes()).hexdigest()
        for sha, passa in ((certo, True), ("0" * 64, False)):
            perfil["head"]["head_sha256"] = sha
            onnx.with_suffix(".edge.json").write_text(json.dumps(perfil), encoding="utf-8")
            m = EdgeAIEngine(OnnxCpuAccelerator(), ClassificationConfig())
            pacote = ModelPackage("coffee-foliar", "1",
                                  hashlib.sha256(onnx.read_bytes()).hexdigest(), onnx)
            if passa:
                m.load(pacote)
                assert m.head is not None and m.profile.head.head_sha256 == certo
            else:
                with pytest.raises(EdgeAIError, match="recalibrado"):
                    m.load(pacote)
        assert TORRE

    def test_sha_da_cabeca_invalido_no_perfil(self) -> None:
        with pytest.raises(ProfileError, match="head_sha256"):
            EdgeProfile.from_dict({**PERFIL, "head": {"encoder_model_id": "x",
                                                      "encoder_sha256": "a" * 64,
                                                      "head_sha256": "xyz"}})


# ---------------------------------------------------------------------------------------------
# No Runtime: o quadro, as vistas e o bloco ``percepcao`` gravados e na fila
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def rig_rgbd(rig_factory, model_repo, clock):
    repo, sha = model_repo
    (repo / f"{sha}.edge.json").write_text(json.dumps(PERFIL), encoding="utf-8")

    def monta(**mancha):
        rig = rig_factory(["verde"], config=RuntimeConfig(sync=SyncConfig(chunk_size=4096)))
        rig.hardware.cameras = [SimulatedRgbdCamera("esq-1000", clock, vaos=[(300, 360)],
                                                    manchas=[{**MANCHA, **mancha}])]
        rig.core.start_mission("m-1", "talhao-1")
        return rig
    return monta


def _resultado(rig) -> dict:
    linhas = rig.core.store.query("SELECT payload FROM records WHERE kind = 'result'")
    assert len(linhas) == 1
    return json.loads(linhas[0]["payload"])


def test_runtime_grava_a_lesao_confirmada(rig_rgbd) -> None:
    rig = rig_rgbd(cor=AMBIGUA, cor_regiao=VERMELHO)
    rig.run(1)
    corpo = _resultado(rig)
    assert corpo["type"] == "POSSIVEL_DOENCA" and corpo["result"]["label"] == "Rust"
    assert corpo["percepcao"]["resolved"] and corpo["percepcao"]["acted"]
    assert corpo["percepcao"]["before"]["category"] == "NORMAL"
    assert corpo["rgbd"]["sufficient"] and corpo["tiles"]["depth_mm"]
    assert corpo["frame"]["view"] == "principal"
    assert not corpo["retention"]["for_validation"]
    papeis = {r["role"] for r in rig.core.store.query("SELECT role FROM files")}
    assert "vista1" not in papeis  # resolvida: as vistas não sobem
    assert rig.core.summary()["percepcao"]["resolvidos"] == 1


def test_runtime_sobe_a_duvida_inteira_para_a_validacao(rig_rgbd) -> None:
    rig = rig_rgbd(cor=AMBIGUA)
    rig.run(1)
    corpo = _resultado(rig)
    assert corpo["retention"]["tier"] == "ORIGINAL"
    assert corpo["retention"]["for_validation"] and corpo["retention"]["sampled_for_validation"]
    arquivos = rig.core.store.query("SELECT role, tier, sha256 FROM files")
    papeis = sorted(r["role"] for r in arquivos)
    assert papeis == ["original", "vista1", "vista2"]
    assert {r["tier"] for r in arquivos} == {"ORIGINAL"}
    assert {f["role"] for f in corpo["files"]} == {"original", "vista1", "vista2"}
    fila = rig.core.store.scalar("SELECT COUNT(*) FROM sync_queue WHERE kind = 'file'")
    assert fila == 3
    for r in arquivos:  # o que vai para a nuvem é a foto, intacta
        caminho = rig.core.store.query("SELECT path FROM files WHERE sha256 = ?",
                                       (r["sha256"],))[0]["path"]
        dados = Path(caminho).read_bytes() if Path(caminho).is_absolute() else (
            Path(rig.data_dir) / caminho).read_bytes()
        with Image.open(io.BytesIO(dados)) as img:
            assert img.size == (1600, 1200)


def test_mapa_de_profundidade_reduzido_na_oak_d(monkeypatch) -> None:
    """A OAK-D com o SDK falso: mapa no quadro, região da exposição pedida e devolvida."""
    from test_rover_v3 import AGORA, Relogio

    from valtexa_rover.hal import rover_v3

    jpeg = io.BytesIO()
    Image.new("RGB", (4056, 3040), (40, 120, 40)).save(jpeg, "JPEG")
    prof = np.full((480, 640), 300, dtype=np.uint16)
    enviados = []

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

        def send(self, x):
            enviados.append(x)

    class Pacote:
        def __init__(self, dados=None, quadro=None):
            self.dados, self.quadro = dados, quadro

        def getData(self):  # noqa: N802
            return np.frombuffer(self.dados, dtype=np.uint8)

        def getFrame(self):  # noqa: N802
            return self.quadro

    class Dispositivo:
        def __init__(self, *_):
            self.filas = {"still": Fila(Pacote(dados=jpeg.getvalue())),
                          "depth": Fila(Pacote(quadro=prof)), "control": Fila()}

        def getOutputQueue(self, nome, **_):  # noqa: N802
            return self.filas[nome]

        def getInputQueue(self, nome):  # noqa: N802
            return self.filas[nome]

    class Controle:
        def __init__(self):
            self.pedidos = {}

        def __getattr__(self, nome):
            return lambda *a: self.pedidos.__setitem__(nome, a)

    class Dai(No):
        Device = Dispositivo
        CameraControl = Controle

    monkeypatch.setattr(rover_v3.time, "sleep", lambda _s: None)
    cam = rover_v3.OakDCamera("esq-1000", "192.168.10.12", 1000.0, clock=Relogio(AGORA),
                              dai=Dai())
    quadro = cam.capture()
    assert quadro.depth is not None and max(quadro.depth.shape) <= 320
    assert quadro.depth.shape == (240, 320) and int(quadro.depth[100, 100]) == 300
    enviados.clear()
    extra = cam.capture_region((0.25, 0.5, 0.75, 1.0), 300.0)
    assert extra.view == "regiao" and extra.depth is not None
    regiao, still, volta = enviados
    assert regiao.pedidos["setAutoExposureRegion"] == (1014, 1520, 2028, 1520)
    assert "setAutoFocusRegion" not in regiao.pedidos  # a Pro W padrão é de foco fixo
    assert "setCaptureStill" in still.pedidos
    assert volta.pedidos["setAutoExposureRegion"] == (0, 0, 4056, 3040)


def test_pedido_de_pausa_por_arquivo(tmp_path) -> None:
    from test_rover_v3 import AGORA, Relogio

    from valtexa_rover.hal.rover_v3 import NavegacaoPorArquivo

    nav = NavegacaoPorArquivo(tmp_path / "navegacao.json", tmp_path / "parar.json",
                              clock=Relogio(AGORA))
    pedido = nav.pedir_pausa(2.0, "percepção ativa: esq-1000")
    gravado = json.loads(nav.pausa_path.read_text(encoding="utf-8"))
    assert gravado == pedido and gravado["seconds"] == 2.0
    assert gravado["pause_until"] == "2026-10-07T12:00:02+00:00"
