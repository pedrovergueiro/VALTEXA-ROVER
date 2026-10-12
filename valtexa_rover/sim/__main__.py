"""``python -m valtexa_rover.sim`` — o simulador do Rover contra uma plataforma no ar.

Usa o protocolo do robô real, por HTTPS (``HttpNetwork``): provisiona com um
token de provisionamento (só na primeira vez; a identidade fica em
``<dados>/identity.json``, 0600), pega a missão **assinada** que a plataforma
despachou, roda com a HAL simulada e sincroniza. Exemplo::

    python -m valtexa_rover.sim --url http://127.0.0.1:8000 --token vtx_prov_... \\
        --serial SIM-0001 --dados /tmp/rover-sim --passos 120 --sem-rede 30-60

O modelo do talhão precisa estar em PRODUÇÃO no registry com o hash do modelo de
demonstração (``valtexa_rover.sim.MODELO_DEMO_SHA256``) — o ``python -m simulator
--rover-runtime`` do ``saas/`` faz esse preparo.

``python -m valtexa_rover.sim fase0 [--saida laudo.md]`` roda a FASE 0 inteira na bancada
**simulada**, sem plataforma, e escreve o laudo (a prévia do robô; o veredito que vale é o
da plataforma — ``python -m simulator --fase0`` no ``saas/``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from valtexa_rover.hal.local import HttpNetwork
from valtexa_rover.runtime.client import DeviceIdentity, provision
from valtexa_rover.sim import CampoSimulado, PlanoDeRede, RoverSimulado


def provisionar(url: str, token: str, serial: str, destino: Path) -> DeviceIdentity:
    """Como o robô real no primeiro boot (``valtexa-rover provisionar``)."""
    return provision(url, token, serial, destino)


def fase0(argv: list[str]) -> int:
    import tempfile

    from valtexa_rover.ensaios.fase0 import laudo_markdown, rodar_na_bancada

    p = argparse.ArgumentParser(prog="valtexa_rover.sim fase0",
                                description="FASE 0 inteira na bancada simulada.")
    p.add_argument("--saida", type=Path, default=Path("laudo-fase0-rover.md"))
    args = p.parse_args(argv)
    print("FASE 0 na bancada SIMULADA (prévia do robô):")
    with tempfile.TemporaryDirectory(prefix="fase0-") as pasta:
        laudos = rodar_na_bancada(Path(pasta), log=print)
    args.saida.write_text(laudo_markdown(laudos, titulo="FASE 0 — laudo do robô (prévia)"),
                          encoding="utf-8")
    print(f"laudo: {args.saida}")
    return 0 if all(x["status"] in ("aprovado", "nao_avaliado") for x in laudos) else 1


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["fase0"]:
        return fase0(argv[1:])
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--url", required=True)
    p.add_argument("--dados", required=True, type=Path)
    p.add_argument("--token", help="token de provisionamento (só na primeira vez)")
    p.add_argument("--serial", default="SIM-0001")
    p.add_argument("--passos", type=int, default=120)
    p.add_argument("--sem-rede", default="", help="janelas sem rede, ex.: 30-60,90-100")
    p.add_argument("--origem", default="-47.0600,-22.9000", help="lon,lat da 1ª planta")
    args = p.parse_args(argv)

    arquivo = args.dados / "identity.json"
    if arquivo.exists():
        identidade = DeviceIdentity.load(arquivo)
    elif args.token:
        identidade = provisionar(args.url, args.token, args.serial, arquivo)
    else:
        print("Sem identidade: passe --token na primeira vez.", file=sys.stderr)
        return 2
    lon, lat = (float(v) for v in args.origem.split(","))
    campo = CampoSimulado.gerar(lon, lat, linhas=3, plantas_por_linha=30,
                                doentes={"L1-P010": "doenca", "L2-P020": "praga",
                                         "L3-P005": "doenca"})
    rover = RoverSimulado(args.dados, identidade, HttpNetwork(args.url), campo)
    missao = rover.core.accept_cloud_mission()
    if missao is None:
        print("Nenhuma missão despachada para este robô.", file=sys.stderr)
        return 3
    janelas = [tuple(int(x) for x in j.split("-")) for j in args.sem_rede.split(",") if j]
    res = rover.rodar(args.passos, PlanoDeRede(sem_rede=janelas))
    rover.core.complete_mission()
    faltando = rover.sincronizar()
    print(json.dumps({"missao": missao, "quadros": res.quadros, "eventos": res.eventos,
                      "passos_sem_rede": res.passos_offline, "pendente": faltando,
                      "simulado": True}, ensure_ascii=False, indent=1))
    return 0 if faltando == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
