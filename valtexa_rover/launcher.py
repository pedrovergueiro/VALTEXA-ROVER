"""Lançador A/B do Rover Runtime (OTA, brief seção 24).

Roda **antes** do Runtime, a partir da instalação de base (que o OTA não troca):

1. conta um boot de cada pacote "em teste" (runtime, Edge AI, configuração);
   passou de ``max_trial_boots`` sem o Runtime confirmar → volta ao anterior;
2. devolve o ``PYTHONPATH`` da versão ativa do Runtime/Edge AI (a pasta do slot).

O Runtime, ao terminar o boot inteiro, confirma (``RoverCore.boot`` →
``OtaManager.confirm_boot``). Se o novo quebra antes disso, o systemd o reinicia,
o lançador conta de novo e, no limite, volta sozinho. Uso no systemd::

    ExecStart=/opt/valtexa/venv/bin/python -m valtexa_rover.launcher \\
        --dados /var/lib/valtexa-rover -- python -m valtexa_rover.cli servico ...
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from valtexa_rover.clock import SystemClock
from valtexa_rover.config import ClassificationConfig
from valtexa_rover.runtime.events import EventEngine
from valtexa_rover.runtime.ota import OtaManager
from valtexa_rover.runtime.store import LocalStore


def prepare(data_dir: Path | str, device_id: str = "", max_trial_boots: int = 2,
            clock=None) -> tuple[list[str], dict[str, str]]:
    """Conta o boot, reverte o que passou do limite e devolve ``(revertidos, ambiente)``."""
    store = LocalStore(data_dir)
    try:
        relogio = clock or SystemClock()
        ota = OtaManager(store, None, EventEngine(store, relogio, device_id), relogio,
                         public_key=None, device_id=device_id, accelerator=None,
                         classification=ClassificationConfig(),
                         max_trial_boots=max_trial_boots)
        revertidos = ota.on_boot()
        caminhos = [str(p) for k in ("edge_ai", "runtime")
                    if (p := ota.active_path(k)) is not None]
    finally:
        store.close()
    ambiente = {}
    if caminhos:
        ambiente["PYTHONPATH"] = os.pathsep.join([*caminhos, os.environ.get("PYTHONPATH", "")])
    return revertidos, ambiente


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - exec do processo
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dados", required=True)
    p.add_argument("--max-boots", type=int, default=2)
    p.add_argument("comando", nargs=argparse.REMAINDER)
    args = p.parse_args(argv)
    revertidos, ambiente = prepare(args.dados, max_trial_boots=args.max_boots)
    if revertidos:
        print(f"OTA: voltou à versão anterior de {', '.join(revertidos)}", file=sys.stderr)
    comando = [c for c in args.comando if c != "--"]
    if not comando:
        return 0
    os.execvpe(comando[0], comando, {**os.environ, **ambiente})
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
