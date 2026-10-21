# valtexa-rover — VALTEXA ROVER RUNTIME

Software do computador embarcado do Rover (Jetson Orin NX, JetPack 6, Python 3.10): HAL,
Edge AI (ONNX Runtime / TensorRT), missão offline-first, buffer circular, retenção em três
níveis, sincronização com a plataforma, OTA, comandos assinados, vídeo ao vivo e ensaios de
validação.

**O que ele não faz:** não diagnostica no sentido final (a Edge AI só **prioriza**: decide o
que é evento e o que sobe em original; quem diagnostica e funde por planta é a IA na nuvem,
VALTEXA-B), não decide aplicação (isso passa pelo agrônomo e pelo Safety Engine do SaaS) e
não move o robô (o movimento é da pilha de navegação/MCU; daqui saem só arquivos de missão,
parada e teleoperação). Não importa o código do SaaS nem o da IA: o que vem da IA está
copiado byte a byte em `valtexa_rover/edge_ai/vendor/` (conferido por `tests/test_vendor.py`).

**Rede:** HTTPS direto pela **Starlink do próprio Rover**. A rede "via base" (rádio da
fazenda) foi arquivada em 20/10/2026 — ver `archive/legacy/radio-fazenda/README.md`.

## Instalar, testar, verificar

```bash
uv venv -p 3.10 && uv pip install -e ".[dev]"     # Python 3.10 = o da Jetson
.venv/bin/pytest && .venv/bin/ruff check .
valtexa-rover config-padrao                        # imprime o rover.json com os padrões
```

No Windows, a suíte também roda pelo venv do SaaS, que instala este pacote em modo
editável (`VALTEXA-SAAS/requirements-dev.txt`). Na Jetson: `deploy/jetson/instalar.sh`, e
na primeira montagem `valtexa-rover diagnostico`, `provisionar` e `preparar-modelo`.

## Documentação

- [`docs/ROVER_RUNTIME.md`](docs/ROVER_RUNTIME.md) — como funciona, como plugar hardware
  real, Rover V3, percepção ativa, orçamento de dados, inspeção adaptativa;
- [`docs/SYNC.md`](docs/SYNC.md) — o contrato de sincronização (fila, chunks, idempotência);
- visão do sistema inteiro (IA, SaaS, Rover), componentes e decisões:
  `VALTEXA-SAAS/docs/ARCHITECTURE.md`, `COMPONENTS.md` e `DECISIONS.md`.

## O que é simulação e o que falta medir

- **Nada rodou no hardware real.** Os adapters de `hal/rover_v3.py` (OAK-D PoE, ZED-F9P,
  Xsens CAN, `tegrastats`, navegação por arquivo) são testados com dados no formato real e
  SDKs falsos; `hal/simulated.py` é simulação e diz que é.
- **Sem pesos reais nos testes**: eles geram um modelo ONNX minúsculo. O tempo de inferência
  na Jetson não foi medido (CPU: ~131 ms por recorte no café).
- O protocolo com a pilha de navegação/MCU (arquivos em `/run/valtexa/`) está definido do
  lado do Runtime, não do lado da navegação.
