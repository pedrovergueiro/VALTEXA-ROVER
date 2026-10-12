"""Retenção e os três níveis de dados — com a medida de volume da seção 13."""

from __future__ import annotations

import io
from datetime import datetime, timedelta

import numpy as np
import pytest
from PIL import Image

from valtexa_rover._compat import UTC
from valtexa_rover.config import (
    Category,
    Priority,
    RetentionConfig,
    RetentionRule,
    RuntimeConfig,
    SyncConfig,
    Tier,
)
from valtexa_rover.runtime.retention import RetentionManager, reduce_image
from valtexa_rover.runtime.store import LocalStore

T0 = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)


def foto(seed: int, size=(1280, 960)) -> bytes:
    """Imagem com textura (folhagem não é cor lisa): o JPEG fica do tamanho de foto real."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (size[1] // 16, size[0] // 16, 3), dtype=np.uint8)
    grande = Image.fromarray(base).resize(size, Image.Resampling.BICUBIC)
    pixels = np.asarray(grande, dtype=np.int16) + rng.integers(-25, 26, (size[1], size[0], 3))
    saida = io.BytesIO()
    Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8)).save(saida, "JPEG", quality=90)
    return saida.getvalue()


def test_decisoes_por_categoria() -> None:
    ret = RetentionManager(RetentionConfig())
    normal = ret.decide(Category.NORMAL, False, T0)
    assert normal.tier is Tier.REDUZIDA and normal.priority is Priority.NORMAL_COMPACTADA
    assert normal.retention_until == T0 + timedelta(days=7)
    for cat in (Category.POSSIVEL_PRAGA, Category.POSSIVEL_DOENCA):
        d = ret.decide(cat, False, T0)
        assert d.tier is Tier.ORIGINAL and d.priority is Priority.DOENCA_PRAGA
        assert d.preserve_sequence
    anomalia = ret.decide(Category.ANOMALIA, False, T0)
    assert anomalia.tier is Tier.ORIGINAL and anomalia.priority is Priority.ANOMALIA
    duvida = ret.decide(Category.NORMAL, True, T0)
    assert duvida.tier is Tier.ORIGINAL


def test_amostra_de_validacao_uma_a_cada_n() -> None:
    ret = RetentionManager(RetentionConfig(normal_sample_every=10))
    tiers = [ret.decide(Category.NORMAL, False, T0) for _ in range(30)]
    originais = [i for i, d in enumerate(tiers) if d.tier is Tier.ORIGINAL]
    assert originais == [9, 19, 29]
    assert all(tiers[i].sampled_for_validation for i in originais)


def test_reducao_nunca_amplia() -> None:
    dados, w, h = reduce_image(foto(1), 320, 60)
    assert max(w, h) == 320 and len(dados) < len(foto(1)) / 5
    pequeno = foto(2, size=(100, 80))
    _d, w2, h2 = reduce_image(pequeno, 320, 60)
    assert (w2, h2) == (100, 80)


def test_volume_10_mil_quadros_com_2_por_cento_de_eventos(rig_factory) -> None:
    """Seção 13: quanto sobe com a política contra enviar tudo em resolução máxima.

    Os bytes são **medidos**: originais JPEG 1280x960 com textura, reduzidas e
    contexto gerados pela mesma função que o Runtime usa, e o tamanho dos
    metadados é o de registros reais produzidos por uma missão do Rig.
    """
    # Tamanho real de um resultado e de um evento, de uma missão de verdade do Runtime.
    rig = rig_factory(["verde"] * 9 + ["vermelho"], auto_sync=False)
    rig.core.start_mission("m-volume", "talhao-1")
    rig.run(10)
    linhas = rig.core.store.query("SELECT kind, LENGTH(payload) AS n FROM records")
    resultado = np.mean([r["n"] for r in linhas if r["kind"] == "result"])
    evento = np.mean([r["n"] for r in linhas if r["kind"] == "event"])

    cfg = RuntimeConfig()
    ret = RetentionManager(cfg.retencao)
    originais = [foto(s) for s in range(4)]
    tamanho_original = [len(o) for o in originais]
    reduzida = [len(reduce_image(o, 320, 60)[0]) for o in originais]
    contexto = [len(reduce_image(o, cfg.buffer.max_side, cfg.buffer.jpeg_quality)[0])
                for o in originais]

    total_quadros, a_cada = 10_000, 50  # 2 % de eventos
    tudo = enviado = 0
    eventos = 0
    for i in range(total_quadros):
        k = i % len(originais)
        tudo += tamanho_original[k]
        e_evento = i % a_cada == a_cada - 1
        decisao = ret.decide(Category.POSSIVEL_DOENCA if e_evento else Category.NORMAL,
                             False, T0)
        enviado += resultado
        if decisao.tier is Tier.ORIGINAL:
            enviado += tamanho_original[k] + evento
            eventos += 1
            if decisao.preserve_sequence:
                enviado += (cfg.buffer.pre_frames + cfg.buffer.post_frames) * contexto[k]
        elif decisao.tier is Tier.REDUZIDA:
            enviado += reduzida[k]

    # Variante "só metadados" para NORMAL (configurável no rover.json).
    so_metadados = enviado - (total_quadros - eventos) * np.mean(reduzida)
    razao = enviado / tudo
    print(
        f"\n[volume] 10.000 quadros, {eventos} eventos ({eventos / 100:.0f}%): "
        f"tudo={tudo / 1e9:.2f} GB, com política={enviado / 1e6:.0f} MB "
        f"({razao:.1%}), NORMAL só metadados={so_metadados / 1e6:.0f} MB "
        f"({so_metadados / tudo:.1%}); original médio={np.mean(tamanho_original) / 1e3:.0f} kB, "
        f"reduzida={np.mean(reduzida) / 1e3:.1f} kB, contexto={np.mean(contexto) / 1e3:.0f} kB, "
        f"resultado={resultado:.0f} B, evento={evento:.0f} B"
    )
    assert eventos == 200
    assert razao < 0.20
    assert so_metadados / tudo < razao


def test_pipeline_real_reduz_o_que_fica_no_disco(rig_factory) -> None:
    """Missão inteira pelo Runtime (Edge AI de verdade sobre o modelo minúsculo)."""
    sequencia = (["verde"] * 49 + ["vermelho"]) * 4  # 200 quadros, 2 % eventos
    rig = rig_factory(sequencia, auto_sync=False,
                      config=RuntimeConfig(sync=SyncConfig(chunk_size=65536)))
    # Imagens maiores, com textura, para a razão fazer sentido.
    for i, arquivo in enumerate(sorted(rig.images.iterdir())):
        cor = arquivo.stem.split("_")[1]
        arquivo.write_bytes(_colorida(cor, i))
    rig.core.start_mission("m-1", "talhao-1")
    rig.run(200)
    originais = sum(p.stat().st_size for p in rig.images.iterdir())
    no_disco = rig.core.store.disk_bytes()
    eventos = rig.core.store.scalar(
        "SELECT COUNT(*) FROM records WHERE kind = 'event' AND type = 'IMAGE_EVENT_DETECTED'")
    print(f"\n[pipeline] 200 quadros/4 eventos: originais={originais / 1e6:.1f} MB, "
          f"guardado={no_disco / 1e6:.2f} MB ({no_disco / originais:.1%})")
    assert eventos == 4
    assert no_disco < originais * 0.25


def _colorida(cor: str, seed: int) -> bytes:
    from conftest import COLORS

    rng = np.random.default_rng(seed)
    pixels = np.zeros((480, 640, 3), dtype=np.int16) + np.array(COLORS[cor])
    pixels += rng.integers(-30, 31, pixels.shape)
    saida = io.BytesIO()
    Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8)).save(saida, "JPEG", quality=90)
    return saida.getvalue()


def test_limpeza_vencidos_e_pressao_de_espaco(tmp_path) -> None:
    store = LocalStore(tmp_path)
    ret = RetentionManager(RetentionConfig())

    def arquivo(nome: str, tier: Tier, uploaded: int, vence: datetime, prioridade: int,
                tamanho: int = 1000) -> None:
        caminho, digest = store.write_blob(nome.encode() * tamanho, ".bin")
        with store.transaction() as conn:
            conn.execute(
                "INSERT INTO files (id, record_id, role, path, sha256, size_bytes, content_type, "
                "tier, retention_until, uploaded, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (nome, "r", "x", caminho, digest, len(nome) * tamanho, "x", tier.value,
                 vence.isoformat(), uploaded, T0.isoformat()),
            )
            conn.execute(
                "INSERT INTO sync_queue (id, kind, ref_id, sha256, size_bytes, priority, status, "
                "created_at, seq) VALUES (?, 'file', ?, ?, 0, ?, ?, ?, 0)",
                (f"q-{nome}", nome, digest, prioridade,
                 "CONCLUIDO" if uploaded else "PENDENTE", T0.isoformat()),
            )

    passado, futuro = T0 - timedelta(days=1), T0 + timedelta(days=30)
    arquivo("A", Tier.REDUZIDA, 1, passado, 5)       # subiu e venceu → sai
    arquivo("B", Tier.REDUZIDA, 0, passado, 5)       # venceu mas não subiu → fica
    arquivo("C", Tier.ORIGINAL, 1, futuro, 2)        # subiu, não venceu
    arquivo("D", Tier.ORIGINAL, 0, futuro, 2)        # evento que não subiu: NUNCA sai
    arquivo("E", Tier.REDUZIDA, 0, futuro, 5)        # normal pendente

    saida = ret.purge(store, T0)
    assert saida["vencidos"] == 1
    restantes = {r["id"] for r in store.query("SELECT id FROM files")}
    assert restantes == {"B", "C", "D", "E"}

    # Pressão: falta muito espaço → apaga o que subiu (C), depois normais pendentes (B, E).
    saida = ret.purge(store, T0, free_bytes=0, min_free_bytes=10**9)
    assert saida["pressao_enviados"] == 1 and saida["pressao_normais"] == 2
    assert {r["id"] for r in store.query("SELECT id FROM files")} == {"D"}
    status = {r["ref_id"]: r["status"] for r in store.query("SELECT * FROM sync_queue")}
    assert status["E"] == "FALHOU" and status["D"] == "PENDENTE"


@pytest.mark.parametrize("tier", [Tier.METADADOS, Tier.REDUZIDA])
def test_regra_normal_configuravel(tier) -> None:
    regras = RetentionConfig().rules | {
        Category.NORMAL: RetentionRule(tier, 3, Priority.NORMAL_COMPACTADA, 256)
    }
    d = RetentionManager(RetentionConfig(rules=regras)).decide(Category.NORMAL, False, T0)
    assert d.tier is tier and d.keeps_image is (tier is not Tier.METADADOS)
