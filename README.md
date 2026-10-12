# valtexa-rover — VALTEXA ROVER RUNTIME

Software do computador embarcado do Rover: HAL, Edge AI (ONNX Runtime), missão
offline-first, buffer circular, retenção em três níveis e sincronização com a plataforma.
Não importa `saas/` nem `ia/`.

```bash
uv venv -p 3.10 && uv pip install -e ".[dev]"
.venv/bin/pytest && .venv/bin/ruff check .
valtexa-rover config-padrao
```

Documentação: [`docs/ROVER_RUNTIME.md`](../docs/ROVER_RUNTIME.md) (como funciona e como
plugar hardware real) e [`docs/SYNC.md`](../docs/SYNC.md) (fila, chunks, idempotência).

**Simulação marcada como simulação**: os adapters de `hal/simulated.py` não falam com
hardware. **Sem pesos reais**: os testes geram um modelo ONNX minúsculo.

**Rover V3 (Jetson Orin NX, JetPack 6, Python 3.10)**: adapters em `hal/rover_v3.py`
(OAK-D PoE, ZED-F9P, Xsens CAN, `tegrastats`, navegação por arquivo), instalador em
`deploy/jetson/instalar.sh` e, para a primeira montagem, `valtexa-rover diagnostico`,
`provisionar` e `preparar-modelo` — ver a seção 14 do `ROVER_RUNTIME.md`. Nenhum adapter
rodou ainda no hardware: os testes usam dados no formato real e SDKs falsos.
