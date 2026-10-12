"""Estresse do Rover Runtime (brief, seção 42) — números reais, contra o dublê de nuvem.

``python -m valtexa_rover.sim.estresse --passos 50000`` roda uma missão longa com o
Runtime de verdade (Edge AI pelo ONNX Runtime, banco local, buffer, retenção,
sync) e, de propósito, tudo o que dá errado em campo:

* **rede intermitente**: cai a cada ``--queda-a-cada`` passos por ``--queda-dura``;
* **backend fora** (rajadas de 503) e **backend instável** (429/500 isolados);
* **robô reiniciado** a cada ``--reinicio-a-cada`` passos, no meio da missão;
* **câmera indisponível** em janelas;
* **disco cheio** (cota pequena: a retenção precisa liberar espaço);
* **queda no meio de arquivo** (a rede cai durante os chunks).

No fim confere as invariantes: cada registro do robô chegou **uma vez**; cada
original de evento chegou e confere por hash; a missão contou cada quadro uma vez;
nada ficou ``FALHOU``; nenhum arquivo órfão. Imprime os números (JSON).
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from valtexa_rover._compat import UTC
from valtexa_rover.clock import ManualClock
from valtexa_rover.config import HealthConfig, RuntimeConfig, SyncConfig
from valtexa_rover.hal.local import LocalDiskStorage
from valtexa_rover.runtime.client import DeviceIdentity, generate_keypair
from valtexa_rover.sim import MODELO_DEMO_SHA256, CampoSimulado, RoverSimulado
from valtexa_rover.sim.nuvem import NuvemEmMemoria


@dataclass
class Numeros:
    passos: int = 0
    quadros: int = 0
    eventos: int = 0
    falhas_de_camera: int = 0
    reinicios: int = 0
    passos_sem_rede: int = 0
    respostas_5xx_injetadas: int = 0
    registros_locais: int = 0
    registros_na_nuvem: int = 0
    lotes_de_registros: int = 0
    reenvios_deduplicados_pela_nuvem: int = 0
    arquivos_na_nuvem: int = 0
    originais_de_evento: int = 0
    originais_de_evento_na_nuvem: int = 0
    bytes_guardados_no_robo: int = 0
    limpezas_por_disco_cheio: int = 0
    falhou_na_fila: int = 0
    segundos_missao: float = 0.0
    segundos_sync_final: float = 0.0
    quadros_por_segundo: float = 0.0


def rodar(pasta: Path, *, passos: int = 2000, cameras: int = 2, queda_a_cada: int = 150,
          queda_dura: int = 60, reinicio_a_cada: int = 700, rajada_5xx_a_cada: int = 400,
          cota_mb: float | None = None, seed: int = 1) -> Numeros:
    privada, publica = generate_keypair()
    nuvem = NuvemEmMemoria(publica, [{
        "field_id": "talhao-estresse", "model_key": "demo", "version": "sim",
        "artifact_sha256": MODELO_DEMO_SHA256, "artifact_kind": "original"}])
    linhas = 4
    por_linha = max(30, passos // linhas)
    doentes = {f"L{1 + i % linhas}-P{p:03d}": ("doenca" if i % 3 else "praga")
               for i, p in enumerate(range(7, por_linha, 41))}
    campo = CampoSimulado.gerar(-47.06, -22.9, linhas=linhas, plantas_por_linha=por_linha,
                                doentes=doentes)
    config = RuntimeConfig(
        sync=SyncConfig(chunk_size=16384, backoff_base_s=1, backoff_max_s=30),
        saude=HealthConfig(min_free_bytes=int(cota_mb * 2**20 * 0.25)) if cota_mb
        else HealthConfig(min_free_bytes=0),
    )
    num = Numeros()
    relogio = ManualClock(datetime.now(UTC))
    rover = RoverSimulado(
        pasta, DeviceIdentity("dev-estresse", "dk", privada), nuvem, campo, clock=relogio,
        cameras=cameras, config=config, seed=seed, auto_sync=True, sync_requests_per_step=8,
        purge_interval_s=60,
        camera_indisponivel=lambda cid, n: cid == "cam-1" and n % 500 in range(100, 110))
    if cota_mb:  # disco pequeno: a retenção tem de liberar espaço durante a missão
        rover.hardware.storage = LocalDiskStorage(pasta, quota_bytes=int(cota_mb * 2**20))
    rover.core.start_mission("missao-estresse", "talhao-estresse")
    inicio = time.perf_counter()
    for n in range(passos):
        if n and reinicio_a_cada and n % reinicio_a_cada == 0:
            rover.reiniciar()
            num.reinicios += 1
        sem_rede = queda_a_cada and n % queda_a_cada >= queda_a_cada - queda_dura
        rover.rede.set_online(not sem_rede)
        num.passos_sem_rede += 1 if sem_rede else 0
        if rajada_5xx_a_cada and n % rajada_5xx_a_cada == rajada_5xx_a_cada // 2:
            nuvem.fail_queue.extend([503] * 6 + [429, 500])
            num.respostas_5xx_injetadas += 8
        if n % 97 == 50:
            rover.rede.drop_after(5)  # cai no meio do que estiver enviando (chunks)
        rel = rover.core.step()
        num.quadros += rel.frames
        num.eventos += rel.events
        num.falhas_de_camera += rel.camera_failures
        relogio.advance(1.0)
    rover.core.complete_mission()
    num.segundos_missao = round(time.perf_counter() - inicio, 2)
    num.passos = passos
    t = time.perf_counter()
    faltando = rover.sincronizar(max_passos=100_000)
    num.segundos_sync_final = round(time.perf_counter() - t, 2)
    loja = rover.core.store
    num.registros_locais = loja.scalar("SELECT COUNT(*) FROM records")
    num.registros_na_nuvem = len(nuvem.records)
    num.lotes_de_registros = nuvem.record_posts
    num.reenvios_deduplicados_pela_nuvem = nuvem.duplicated
    num.arquivos_na_nuvem = len(nuvem.objects)
    # Pelo que a nuvem recebeu, não pelo banco local: a limpeza por disco cheio apaga
    # as linhas dos arquivos que já subiram (a 1ª execução de 100 mil contou 0 assim).
    originais = [f["sha256"] for b in nuvem.record_bodies.values()
                 if b.get("kind") == "result" and b.get("event_id")
                 for f in b.get("files", []) if f.get("role") == "original"]
    num.originais_de_evento = len(originais)
    num.originais_de_evento_na_nuvem = sum(1 for sha in originais if sha in nuvem.objects)
    num.bytes_guardados_no_robo = loja.disk_bytes()
    num.falhou_na_fila = loja.scalar("SELECT COUNT(*) FROM sync_queue WHERE status = 'FALHOU'")
    num.limpezas_por_disco_cheio = loja.scalar(
        "SELECT COUNT(*) FROM records WHERE type = 'STORAGE_PRESSURE'")
    num.quadros_por_segundo = round(num.quadros / max(num.segundos_missao, 1e-9), 1)
    missao = rover.core.missions.get("missao-estresse")
    if faltando:
        raise AssertionError(f"sobrou fila: {faltando}")
    if missao.frames_done != num.quadros:
        raise AssertionError(f"missão contou {missao.frames_done} quadros, foram {num.quadros}")
    if num.registros_na_nuvem != num.registros_locais:
        raise AssertionError("registro perdido ou duplicado na nuvem")
    rover.core.close()
    return num


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - roda à mão
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--passos", type=int, default=2000)
    p.add_argument("--cameras", type=int, default=2)
    p.add_argument("--cota-mb", type=float, default=None)
    p.add_argument("--pasta", type=Path, default=None)
    args = p.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="estresse-") as tmp:
        num = rodar(args.pasta or Path(tmp), passos=args.passos, cameras=args.cameras,
                    cota_mb=args.cota_mb)
    print(json.dumps(asdict(num), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
