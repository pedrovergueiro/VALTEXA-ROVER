"""Linha de comando de serviço do Rover (técnico de campo).

* ``valtexa-rover config-padrao`` — imprime o ``rover.json`` com todos os padrões;
* ``valtexa-rover validar-config rover.json`` — confere o arquivo (chave errada é erro);
* ``valtexa-rover status --dados /var/lib/valtexa-rover`` — missão em curso, fila de
  sincronização por status e prioridade, modelo ativo. Só lê o banco local;
* ``valtexa-rover provisionar --url ... --serial ... --identidade ...`` — primeiro boot: gera
  a chave **no robô** e registra só a pública (token pela entrada padrão ou
  ``VALTEXA_TOKEN_PROVISIONAMENTO``, nunca na linha de comando);
* ``valtexa-rover preparar-modelo modelo.onnx --dados ...`` — carrega o modelo e o perfil
  neste computador, mede um quadro e copia para a área de staging do OTA (``<sha256>.onnx``);
* ``valtexa-rover diagnostico --dados ... [--hal modulo:funcao] [--salvar pasta]`` — primeira
  montagem: lê cada peça do hardware uma vez (câmeras, GNSS, IMU, acelerador, disco) e diz
  qual falhou, sem identidade nem rede;
* ``valtexa-rover servico --dados ... --identidade ... --config ... --hal modulo:funcao
  --url https://...`` — **o Runtime em serviço** (systemd): boot, pega a missão assinada
  da plataforma, laço de captura, conclui quando o controlador diz que a rota acabou,
  sincroniza, OTA fora de missão. ``--hal`` aponta a função que monta a
  :class:`~valtexa_rover.hal.interfaces.Hardware` com os adapters do robô
  (``funcao(dados=Path, url=str, clock=Clock) -> Hardware``); para demonstração,
  ``valtexa_rover.sim:hal_de_demonstracao`` (SIMULADO). Ver ``docs/ROVER_RUNTIME.md``;
* ``valtexa-rover enlace-chave --identidade ... --chave ...`` — rádio da fazenda: cria (se
  não existir) a chave X25519 do enlace **no robô** e imprime a pública assinada pela
  identidade, para cadastrar na base (``docs/COMUNICACAO_FAZENDA.md``).
"""

from __future__ import annotations

import argparse
import json
import sys

from valtexa_rover import EDGE_AI_VERSION, RUNTIME_VERSION
from valtexa_rover.config import ConfigError, RuntimeConfig
from valtexa_rover.runtime.store import LocalStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="valtexa-rover", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="comando", required=True)
    sub.add_parser("config-padrao", help="imprime a configuração padrão")
    validar = sub.add_parser("validar-config", help="confere um rover.json")
    validar.add_argument("arquivo")
    status = sub.add_parser("status", help="estado do banco local")
    status.add_argument("--dados", required=True)
    servico = sub.add_parser("servico", help="roda o Runtime (systemd)")
    servico.add_argument("--dados", required=True)
    servico.add_argument("--identidade", required=True)
    servico.add_argument("--config", default=None)
    servico.add_argument("--hal", required=True, help="modulo:funcao que monta o Hardware")
    servico.add_argument("--url", required=True)
    servico.add_argument("--intervalo", type=float, default=0.5, help="segundos entre ciclos")
    servico.add_argument("--busca-missao", type=float, default=15.0,
                         help="segundos entre perguntas por missão nova")
    servico.add_argument("--busca-comando", type=float, default=2.0,
                         help="segundos entre buscas de comando do centro de comando")
    servico.add_argument("--max-ciclos", type=int, default=None, help=argparse.SUPPRESS)
    prov = sub.add_parser("provisionar", help="primeiro boot: registra o robô na plataforma")
    prov.add_argument("--url", required=True)
    prov.add_argument("--serial", required=True)
    prov.add_argument("--identidade", required=True)
    prep = sub.add_parser("preparar-modelo", help="confere e põe um modelo no staging do OTA")
    prep.add_argument("onnx")
    prep.add_argument("--dados", required=True)
    prep.add_argument("--modelo", default=None, help="model_key (padrão: nome do arquivo)")
    prep.add_argument("--sem-perfil", action="store_true",
                      help="aceita modelo sem .edge.json (classifica o quadro inteiro)")
    prep.add_argument("--encoder", default=None,
                      help="torre de visão ONNX (framing_vision.onnx) dos modelos com cabeça")
    prep.add_argument("--quadros", type=int, default=3,
                      help="quadros sintéticos de 4056x3040 para medir (0 = não mede)")
    diag = sub.add_parser("diagnostico", help="lê cada peça do hardware uma vez")
    diag.add_argument("--dados", required=True)
    diag.add_argument("--hal", default="valtexa_rover.hal.rover_v3:hal_rover_v3")
    diag.add_argument("--url", default="https://diagnostico.invalid", help=argparse.SUPPRESS)
    diag.add_argument("--salvar", default=None, help="pasta para os quadros capturados")
    diag.add_argument("--espera-gnss", type=float, default=5.0,
                      help="segundos esperando o primeiro fix")
    ens = sub.add_parser("ensaio", help="roda um ensaio de validação aqui, sem plataforma")
    ens.add_argument("nome", help="cameras, profundidade, gnss, imu, parada, homem_morto, "
                                  "energia (calor e rede: pelo centro de comando)")
    ens.add_argument("--dados", required=True)
    ens.add_argument("--hal", default="valtexa_rover.hal.rover_v3:hal_rover_v3")
    ens.add_argument("--url", default="https://ensaio.invalid", help=argparse.SUPPRESS)
    ens.add_argument("--param", action="append", default=[], metavar="CHAVE=VALOR",
                     help="parâmetro do ensaio (valor em JSON: 500, true, [0.4,0.4,0.2,0.2])")
    chave = sub.add_parser("enlace-chave", help="pública do enlace da fazenda, assinada")
    chave.add_argument("--identidade", required=True)
    chave.add_argument("--chave", required=True, help="ex.: /var/lib/valtexa-rover/enlace.key")
    chave.add_argument("--publicar", metavar="URL",
                       help="publica na plataforma (HTTPS direto, na implantação): a base da "
                            "fazenda busca de lá, sem copiar arquivo à mão")
    args = parser.parse_args(argv)
    if args.comando == "enlace-chave":
        return enlace_chave(args)
    if args.comando == "ensaio":
        return ensaio_local(args)
    if args.comando == "servico":
        return rodar_servico(args)
    if args.comando == "provisionar":
        return provisionar(args)
    if args.comando == "preparar-modelo":
        return preparar_modelo(args)
    if args.comando == "diagnostico":
        return diagnostico(args)

    if args.comando == "config-padrao":
        print(json.dumps(RuntimeConfig().to_dict(), indent=2, ensure_ascii=False))
        return 0
    if args.comando == "validar-config":
        try:
            RuntimeConfig.load(args.arquivo)
        except ConfigError as exc:
            print(f"inválido: {exc}", file=sys.stderr)
            return 1
        print("ok")
        return 0

    store = LocalStore(args.dados)
    try:
        fila = store.query(
            "SELECT status, priority, COUNT(*) AS n, COALESCE(SUM(size_bytes), 0) AS bytes "
            "FROM sync_queue GROUP BY status, priority ORDER BY status, priority"
        )
        missao = store.query(
            "SELECT id, field_id, state, frames_done, events, model_key, model_version "
            "FROM missions ORDER BY started_at DESC LIMIT 1"
        )
        saida = {
            "runtime_version": RUNTIME_VERSION,
            "edge_ai_version": EDGE_AI_VERSION,
            "active_model": store.get("active_model"),
            "last_mission": dict(missao[0]) if missao else None,
            "queue": [dict(linha) for linha in fila],
            "telemetry_unsent": store.scalar("SELECT COUNT(*) FROM telemetry WHERE sent = 0"),
            "sync_blocked": bool(store.get("sync_blocked", False)),
        }
    finally:
        store.close()
    print(json.dumps(saida, indent=2, ensure_ascii=False))
    return 0


def diagnostico(args) -> int:
    """Uma leitura de cada peça; ``ok: false`` com o erro em vez de parar no primeiro."""
    import importlib
    from pathlib import Path

    from valtexa_rover.clock import SystemClock

    modulo, _, funcao = args.hal.partition(":")
    relogio = SystemClock()
    try:
        hardware = getattr(importlib.import_module(modulo), funcao)(
            dados=Path(args.dados), url=args.url, clock=relogio)
    except Exception as exc:  # hardware.json ausente, SDK faltando, peça que não abre
        print(json.dumps({"ok": False, "montagem": f"{type(exc).__name__}: {exc}"},
                         ensure_ascii=False))
        return 1
    from valtexa_rover.hal import diagnostico as diag

    laudo = diag.run(hardware, salvar=Path(args.salvar) if args.salvar else None,
                     espera_gnss=args.espera_gnss)
    print(json.dumps(laudo, indent=2, ensure_ascii=False, default=str))
    return 0 if laudo["ok"] else 1


def ensaio_local(args) -> int:
    """Um ensaio no terminal da Jetson: o laudo (prévia do robô) em JSON, com progresso."""
    import contextlib
    import importlib
    from pathlib import Path

    from valtexa_rover.clock import SystemClock
    from valtexa_rover.ensaios import CATALOGO, Contexto, rodar_por_nome
    from valtexa_rover.ensaios import _TermicaDoMonitor as Termica
    from valtexa_rover.ensaios.energia import SensoresComoEnergia

    if args.nome not in CATALOGO:
        print(f"ensaio {args.nome!r}: use {', '.join(CATALOGO)}", file=sys.stderr)
        return 1
    params = {}
    for item in args.param:
        chave, _, valor = item.partition("=")
        try:
            params[chave] = json.loads(valor)
        except ValueError:
            params[chave] = valor
    modulo, _, funcao = args.hal.partition(":")
    relogio = SystemClock()
    try:
        hardware = getattr(importlib.import_module(modulo), funcao)(
            dados=Path(args.dados), url=args.url, clock=relogio)
    except Exception as exc:  # hardware.json ausente, SDK faltando, peça que não abre
        print(json.dumps({"ok": False, "montagem": f"{type(exc).__name__}: {exc}"},
                         ensure_ascii=False))
        return 1
    fontes = {}
    servico = getattr(hardware.network, "servico", None)
    if servico is not None:
        # Rede via base: os ensaios de enlace falam com a base. Com o serviço do Runtime
        # rodando, os dois brigariam pelas portas UDP — pare o serviço antes, ou mande o
        # ensaio pelo console (comando assinado).
        from valtexa_rover.ensaios.enlace import FonteDoEnlace

        fontes["enlace"] = FonteDoEnlace(servico)
    if hardware.system is not None:
        fontes["termica"] = Termica(hardware.system)
    if hardware.sensors is not None:
        with contextlib.suppress(Exception):
            if "battery_percent" in hardware.sensors.read().values:
                fontes["energia"] = SensoresComoEnergia(hardware.sensors)

    def progresso(fracao: float, etapa: str, parcial: dict | None) -> None:
        print(f"[{fracao:5.0%}] {etapa} {json.dumps(parcial or {}, ensure_ascii=False)}",
              file=sys.stderr)

    laudo = rodar_por_nome(args.nome, params, Contexto(hardware, relogio, fontes, progresso))
    print(json.dumps(laudo, indent=2, ensure_ascii=False, default=str))
    return 0 if laudo["status"] in ("aprovado", "nao_avaliado") else 1


def provisionar(args) -> int:
    import os

    from valtexa_rover.runtime.client import provision

    token = os.environ.get("VALTEXA_TOKEN_PROVISIONAMENTO") or sys.stdin.readline().strip()
    if not token:
        print("sem token: passe pela entrada padrão ou VALTEXA_TOKEN_PROVISIONAMENTO",
              file=sys.stderr)
        return 2
    try:
        ident = provision(args.url, token, args.serial, args.identidade)
    except (FileExistsError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps({"device_id": ident.device_id, "key_id": ident.key_id,
                      "identidade": args.identidade}, ensure_ascii=False))
    return 0


def preparar_modelo(args) -> int:
    import io
    import time
    from pathlib import Path

    import numpy as np
    from PIL import Image

    from valtexa_rover.config import ClassificationConfig
    from valtexa_rover.edge_ai.engine import EdgeAIEngine
    from valtexa_rover.hal.rover_v3 import JetsonAccelerator
    from valtexa_rover.runtime.model_manager import ModelError, stage_model

    acelerador = JetsonAccelerator()  # TensorRT > CUDA > CPU, o que houver aqui
    motor = EdgeAIEngine(acelerador, ClassificationConfig())
    try:
        saida = stage_model(args.onnx, Path(args.dados) / "ota" / "models", motor,
                            model_key=args.modelo, require_profile=not args.sem_perfil,
                            encoder=args.encoder)
    except ModelError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    saida["providers"] = acelerador.providers()
    if args.quadros > 0:
        # Folhagem sintética no tamanho da OAK-D (12 MP), a 270 mm: mede o recorte + a rede.
        rng = np.random.default_rng(0)
        rgb = (rng.random((3040, 4056, 3)) * [60, 160, 60]).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, "JPEG", quality=90)
        quadro = buf.getvalue()
        motor.infer(quadro, distance_mm=270.0, camera_height_mm=1000.0)  # aquece (TensorRT)
        tempos, recortes = [], 0
        for _ in range(args.quadros):
            t0 = time.perf_counter()
            r = motor.infer(quadro, distance_mm=270.0, camera_height_mm=1000.0)
            tempos.append(time.perf_counter() - t0)
            recortes = len(r.tiles)
        saida["medicao"] = {"quadro_px": [4056, 3040], "distancia_mm": 270.0,
                            "recortes": recortes,
                            "ms_por_quadro": round(1000 * sorted(tempos)[len(tempos) // 2], 1)}
    print(json.dumps(saida, indent=2, ensure_ascii=False))
    return 0


def enlace_chave(args) -> int:
    from valtexa_rover.enlace.cripto import ChaveDeEnlace, publica_assinada
    from valtexa_rover.runtime.client import DeviceIdentity

    identidade = DeviceIdentity.load(args.identidade)
    k = ChaveDeEnlace.carregar_ou_criar(args.chave)
    doc = publica_assinada(k, identidade.device_id, identidade.private_key)
    print(json.dumps(doc))
    if args.publicar:
        from valtexa_rover.clock import SystemClock
        from valtexa_rover.hal.interfaces import NetworkUnavailableError
        from valtexa_rover.hal.local import HttpNetwork
        from valtexa_rover.runtime.client import DeviceClient

        cliente = DeviceClient(identidade, HttpNetwork(args.publicar), SystemClock())
        try:
            r = cliente.request("PUT", "/v1/device/enlace/chave", json_body=doc)
        except NetworkUnavailableError as exc:
            print(f"sem rede até a plataforma: {exc}", file=sys.stderr)
            return 1
        if r.status != 200:
            print(f"plataforma recusou: HTTP {r.status} {r.body[:200]!r}", file=sys.stderr)
            return 1
        print("publicada na plataforma: a base da fazenda busca de lá", file=sys.stderr)
    return 0


def rodar_servico(args) -> int:
    """O laço do robô. SIGTERM (systemd) para com a transação do quadro fechada."""
    import contextlib
    import importlib
    import signal
    import threading
    from pathlib import Path

    from valtexa_rover.clock import SystemClock
    from valtexa_rover.hal.interfaces import NetworkUnavailableError
    from valtexa_rover.runtime.client import DeviceIdentity
    from valtexa_rover.runtime.core import RoverCore
    from valtexa_rover.runtime.mission import MissionError
    from valtexa_rover.runtime.model_manager import FolderArtifactSource, ModelError

    modulo, _, funcao = args.hal.partition(":")
    fabrica = getattr(importlib.import_module(modulo), funcao)
    dados = Path(args.dados)
    relogio = SystemClock()
    hardware = fabrica(dados=dados, url=args.url, clock=relogio)
    core = RoverCore(hardware, dados, DeviceIdentity.load(args.identidade),
                     config=RuntimeConfig.load(args.config), clock=relogio,
                     artifact_source=FolderArtifactSource(dados / "ota" / "models"),
                     config_path=args.config)
    core.boot()
    # Rede via base (rádio da fazenda): a política de perda de enlace age na navegação.
    enlace = getattr(hardware.network, "servico", None)
    if enlace is not None:
        enlace.liga(hardware.controller)
    parar = threading.Event()
    for sinal in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(ValueError):  # fora da thread principal (testes)
            signal.signal(sinal, lambda *_: parar.set())
    from valtexa_rover.runtime.comandos import CommandExecutor

    comandos = CommandExecutor(core, args.config)
    from valtexa_rover.runtime.client import DeviceClient
    from valtexa_rover.runtime.video import TransmissorDeVideo

    # Cliente próprio (mesma identidade, nonce próprio por pedido): o vídeo corre numa
    # thread e nunca segura o laço da missão.
    video = TransmissorDeVideo(
        hardware, DeviceClient(core.identity, hardware.network, relogio), relogio,
        missao_em_curso=lambda: core.missions.current() is not None,
        ocupado=lambda: comandos.ensaios.ocupado is not None,
        quadro_recente=core.ultimos_quadros.get).iniciar()
    comandos.video = video
    ultima_busca = ultimo_comando = -1e9
    anunciada: str | None = None
    ciclos = 0
    codigo = 0
    while not parar.is_set():
        agora = relogio.monotonic()
        if agora - ultimo_comando >= args.busca_comando:
            ultimo_comando = agora
            comandos.poll()  # centro de comando: testes, foto, modo, config, modelo
            if comandos.restart_requested:
                core.log.info("reiniciando a pedido do centro de comando")
                codigo = 75  # o systemd (Restart=always) sobe de novo
                break
        comandos.teleop()  # só age em modo manual; sem ponto válido, zero
        # Com ensaio rodando, nenhuma missão começa (as peças são do ensaio).
        if (core.missions.current() is None and comandos.ensaios.ocupado is None
                and agora - ultima_busca >= args.busca_missao):
            ultima_busca = agora
            try:
                core.accept_cloud_mission()
            except (NetworkUnavailableError, MissionError, ModelError) as exc:
                core.log.warning("missão não aceita", extra={"data": {"motivo": str(exc)}})
        core.step()
        missao = core.missions.current()
        if enlace is not None:
            enlace.missao_atual = None if missao is None else {"mission_id": missao.id,
                                                               **(missao.payload or {})}
            # o mapa de cobertura do rádio vem assinado na missão (último levantamento)
            enlace.cobertura_da_missao(
                None if missao is None
                else ((missao.payload or {}).get("command") or {}).get("cobertura"))
            travessia = getattr(enlace, "travessia", None)
            if travessia is not None:
                enlace.sync_pendente = core.sync.pending()
                if travessia.pede_sincronizar():
                    # antes da zona morta (sobe o que dá) e na volta (sincroniza tudo já)
                    with contextlib.suppress(NetworkUnavailableError):
                        core.sync.step(500)
            if hardware.gps is not None:
                # posição para o levantamento de cobertura e a missão ciente do enlace;
                # GNSS fora não para o laço (o enlace segue sem posição)
                with contextlib.suppress(Exception):
                    enlace.atualiza_posicao(hardware.gps.read())
        anunciar = getattr(hardware.controller, "announce_mission", None)
        if anunciar is not None and (missao and missao.id) != anunciada:
            anunciar(missao)  # a navegação recebe a missão (e a rota) ou None
            anunciada = missao and missao.id
        controle = hardware.controller.state() if hardware.controller else {}
        # Um route_done de outra missão (arquivo que sobrou) não encerra esta.
        if (missao is not None and controle.get("route_done")
                and controle.get("mission_id", missao.id) == missao.id):
            core.complete_mission()
        ciclos += 1
        if args.max_ciclos is not None and ciclos >= args.max_ciclos:
            break
        parar.wait(args.intervalo)
    video.parar()
    core.close()
    if enlace is not None:
        enlace.parar()
    return codigo


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
