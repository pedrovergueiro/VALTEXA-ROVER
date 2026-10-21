# ARQUIVADO — rádio da fazenda (rede do Rover "via base")

**Arquivado em 20/10/2026.** Não é importado pelo Runtime, não é testado, não entra no pacote
(`pyproject.toml` inclui só `valtexa_rover*`) nem no lint (`extend-exclude = ["archive"]`).

## Por quê

A arquitetura de comunicação mudou: **não existe mais base da fazenda nem rádio**. Cada Rover
tem a própria Starlink e fala com a plataforma por HTTPS direto
(`valtexa_rover.hal.local.HttpNetwork`). A rede "via base" sempre foi opcional
(`"rede": {"modo": "base"}` no `hardware.json`) e nunca rodou em hardware — só contra o
emulador e UDP no `localhost`.

## O que está aqui

| Caminho | Era | O que fazia |
|---|---|---|
| `valtexa_rover/enlace/` | `valtexa_rover/enlace/` | Protocolo Rover ⇄ base: quadro, cripto X25519/ChaCha20, filas SQLite, nó ARQ com prioridades, RPC (HTTP sobre o enlace), malha com repetidoras, levantamento de cobertura, travessia de zona morta, emulador, bancadas, ensaio de 24 h |
| `valtexa_rover/hal/rede_base.py` | `valtexa_rover/hal/rede_base.py` | `ServicoDeEnlace` / `rede_via_base`: a HAL de rede "via base" |
| `valtexa_rover/ensaios/enlace.py` | `valtexa_rover/ensaios/enlace.py` | Ensaios da FASE ENLACE (`enlace_rssi`, `enlace_latencia`, `enlace_troca`, `enlace_starlink`, `enlace_energia_base`, `enlace_travessia`) e o antigo `relogio` com dois lados |
| `tests/` | `tests/test_enlace*.py`, `test_malha.py`, `test_travessia.py`, `test_ensaio24h.py` + o teste de slot de vídeo que estava em `tests/test_video.py` | 67 testes (passavam em 20/10/2026, 366 no total do Rover antes do arquivamento) |
| `deploy/hardware.rover-v3.fazenda.json.exemplo` | `deploy/` | Exemplo de `hardware.json` com `"rede": {"modo": "base", ...}` |

## O que NÃO foi arquivado (continua no produto)

- **`valtexa_rover/hal/relogio.py`** (era `enlace/relogio.py`): leitura do chrony/GNSS. O
  relógio importa com Starlink também (assinatura das requisições, casar foto/GNSS/IMU).
- **Ensaio `relogio`** (`valtexa_rover/ensaios/relogio.py`): agora só o lado do Rover, na
  FASE 0.

## Substituído por

HTTPS direto pela Starlink de cada Rover (`HttpNetwork`), com a fila offline-first do
Runtime (`runtime/sync.py`), o orçamento de dados (`runtime/orcamento.py`) e a política de
retenção. Um `hardware.json` antigo com `"rede": {"modo": "base"}` agora falha no boot com
mensagem clara (`hal/rover_v3.py`).

## Dependências conhecidas

- **VALTEXA-BASE** (o software da base) importava `valtexa_rover.enlace`: foi arquivado junto
  (ver `VALTEXA-BASE/README.md`).
- **VALTEXA-SAAS**: as rotas de chave de enlace, pares da base, cobertura do rádio e a tela
  Bases foram arquivadas em `VALTEXA-SAAS/archive/legacy/radio-fazenda/`. As colunas do banco
  (`device.enlace_chave`, `field.cobertura`, migrações 0030 e 0031) **ficam**: migração
  aplicada não se reescreve.
- Documentação: `VALTEXA-BASE/docs/COMUNICACAO_FAZENDA.md`.

## Como recuperar

1. Copie `valtexa_rover/enlace/`, `valtexa_rover/hal/rede_base.py` e
   `valtexa_rover/ensaios/enlace.py` de volta para os mesmos caminhos; traga
   `hal/relogio.py` de volta para `enlace/relogio.py` (ou ajuste o import em `ensaios/enlace.py`).
2. Em `valtexa_rover/ensaios/__init__.py`, volte a importar `ENSAIOS_DE_ENLACE` e
   `FonteDoEnlace` e a registrar a fonte `enlace` em `fontes_do_runtime`.
3. Em `hal/rover_v3.py`, volte o ramo `rede_cfg.get("modo") == "base"` → `rede_via_base`.
4. Em `cli.py`, volte o comando `enlace-chave` e o trecho do laço do serviço que liga o
   serviço de enlace (missão, cobertura, travessia, posição).
5. Copie os testes de `tests/` de volta para `tests/`.

O estado exato antes do arquivamento também está no snapshot
`C:\Users\pedro\valtexa_snapshots\2026-10-20_antes-da-consolidacao\VALTEXA-ROVER.tar.gz`
(e, para os arquivos já commitados, no commit `ece8c65`).
