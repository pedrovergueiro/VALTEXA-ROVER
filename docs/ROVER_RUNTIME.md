# VALTEXA ROVER RUNTIME (`rover/`)

O software que roda no computador embarcado do Rover (brief, seções 11, 12, 14–17, 20–22,
40, 53 e 54). Pacote Python próprio, **separado** da plataforma (`saas/`) e da IA de nuvem
(`ia/`): não importa nenhum dos dois, instala-se sozinho e roda sem internet.

> **O que é real e o que é simulado (seção 55).** Real e testado: Edge AI pelo ONNX
> Runtime, banco local transacional, buffer circular, retenção, fila de sincronização,
> Model Manager com hash, missão assinada, telemetria/saúde. **Simulado e marcado como
> simulação**: câmera (pasta de imagens), GPS (trajetória), IMU, sensores, controlador e a
> rede que cai e volta (`hal/simulated.py`; todo adapter diz `simulated=True`, e cada
> resultado grava `hardware.simulated`). **Sem pesos reais**: os testes geram um modelo
> ONNX minúsculo (média RGB → 3x4 → softmax). Desempenho do modelo real: a verificar.

> **Documento atual do Rover Runtime** (veio de `VALTEXA-SAAS/docs/plataforma/` em
> 20/10/2026). Visão do sistema inteiro: `VALTEXA-SAAS/docs/ARCHITECTURE.md`. As seções
> de rodadas antigas citam caminhos do monorepo da nuvem (`rover/...`, `saas/...`): no
> espaço de trabalho são `VALTEXA-ROVER/...` e `VALTEXA-SAAS/...`.

## 1. Instalar e testar

```bash
cd rover
uv venv -p 3.11 && uv pip install -e ".[dev]"
.venv/bin/pytest            # 95 testes, ~25 s (inclui estresse em tamanho de CI)
.venv/bin/ruff check .
```

Dependências de execução: `numpy`, `pillow`, `onnxruntime`, `cryptography` (Ed25519 — a
stdlib não tem). Rede é `urllib` e o banco é `sqlite3`, ambos da stdlib. `onnx` só nos
testes (para gerar o modelo minúsculo).

## 2. Mapa do pacote

| Seção do brief | Módulo | O que faz |
| --- | --- | --- |
| 20, 53 HAL | `hal/interfaces.py` | contratos: `Camera`, `Gps`, `Imu`, `Sensors`, `StorageDevice`, `Network`, `AiAccelerator`, `Controller`; `Hardware` junta os adapters |
| | `hal/local.py` | adapters **reais** genéricos: disco local (com cota), HTTPS (`urllib`), *providers* do ONNX Runtime |
| | `hal/simulated.py` | adapters **SIMULADOS**: `FolderCamera`, `TrajectoryGps`, `SimulatedImu`, `SimulatedSensors`, `FlakyNetwork`, `SimulatedController` |
| 11, 14, 22 Edge AI | `edge_ai/engine.py` | pré-processamento → ONNX → softmax → rótulo → categoria (NORMAL / POSSÍVEL PRAGA / POSSÍVEL DOENÇA / ANOMALIA) + baixa confiança |
| | `edge_ai/vendor/` | `preprocess.py` e `onnx_runtime.py` **copiados byte a byte** do `ia/`; `tests/test_vendor.py` compara os SHA-256 |
| 21 Configuration | `config.py` | `rover.json`; chave desconhecida é erro |
| 21 Core | `runtime/core.py` | o laço quadro a quadro e o ciclo de vida (boot, missão, conclusão) |
| 21 Mission Manager | `runtime/mission.py` | missão no banco local; conferência do comando assinado |
| 21, 22 Model Manager | `runtime/model_manager.py` | só modelo do registry, hash conferido antes de carregar |
| 40 Event Engine | `runtime/events.py` | eventos com ID e hash, direto na fila |
| 12 Local Storage | `runtime/store.py` | SQLite WAL + `synchronous=FULL`; arquivos por hash, atômicos |
| 15 Buffer circular | `runtime/buffer.py` | N antes + evento + M depois, por câmera |
| 13, 16, 17 Retenção | `runtime/retention.py` | decide o nível de cada quadro; limpa o que já subiu |
| 18, 19, 52 Sync | `runtime/sync.py` | fila priorizada, chunks, retomada, *backoff* — ver `docs/SYNC.md` |
| 21 Telemetry, Health | `runtime/health.py` | horímetro, disco, temperatura, FPS/latência, falhas ativas |
| 21 Logging | `runtime/logs.py` | JSON por linha, arquivo rotativo |
| — | `runtime/client.py` | identidade do robô e requisições assinadas (Ed25519) |
| 24 OTA Manager | `runtime/ota.py`, `launcher.py` | pacote assinado (Ed25519), download com retomada, hash por arquivo, instalação lado a lado, saúde, troca atômica, confirmação no boot (A/B) e *rollback* — ver `docs/OTA.md` |
| 41 Simulador | `sim/` | o **mesmo** Runtime sobre a HAL simulada: campo com espaçamento, câmera de cena, rede que cai, reinício; `sim/estresse.py`; `sim/nuvem.py` (dublê) |
| — | `cli.py` | `valtexa-rover config-padrao | validar-config | status | servico` |
| ~~Fazenda (P12)~~ | ~~`enlace/`, `hal/rede_base.py`~~ | **arquivado em 20/10/2026** (`archive/legacy/radio-fazenda/`): a rede "via base" por rádio — ver §19 |

**Ainda fora:** Security do disco (cifra — LUKS/dm-crypt no sistema operacional, fora do
pacote) e *secure boot* (dependem do computador embarcado escolhido); runtime de drone.

## 3. O pipeline de cada quadro (seção 14)

```
câmera → captura → pré-processamento → Edge AI → categoria → decisão de retenção
       → disco (arquivo atômico) → transação: resultado + arquivos + evento + contexto
         do buffer + itens da fila + progresso da missão → (fila) → sincronização
```

1. **Captura** pela HAL. Câmera que falha vira falha ativa (`camera:<id>`) e um
   `HARDWARE_FAULT` (um só, não um por quadro); a missão segue com as outras câmeras.
2. **Edge AI**: o pré-processamento é o do treino (sem EXIF, *resize* bilinear do
   TensorFlow reimplementado em NumPy, escala para `[-1, 1]`). Inferência pelo ONNX Runtime
   com os *providers* do acelerador. Categoria pelo `<modelo>.categories.json` (recomendado)
   ou pelo nome do rótulo.
   * **Com o perfil `<modelo>.edge.json`** (o caso dos modelos reais, gerado por
     `python -m training.fit_edge fit` em VALTEXA-B): o quadro é cortado em **recortes de
     150 mm** no tamanho físico, pela distância medida (mediana da profundidade da OAK-D)
     e pelo campo de visão — o modelo foi treinado com folha enchendo o recorte, e uma
     lesão de Sigatoka some num quadro de 1 m reduzido a 224 px. O quadro vira evento pela
     **evidência** (`evidence` do perfil): cada recorte vira a log-razão de
     verossimilhança da faixa da sua probabilidade, por doença; o escore é a maior, entre
     as doenças, da média nos recortes com o desconto de correlação da fusão; evento
     acima de `event_evidence`, `baixa_confianca` acima de `near_miss_evidence` (original
     guardado para validação); confiança média abaixo de `anomaly_mean_confidence_below`
     → ANOMALIA. Contar recortes acima de um limiar não serve: o modelo põe ≥ 0,99 de
     doença em pedaço de folha sadia (na banana, 33% dos quadros sadios virariam evento).
     Tabelas e limiares medidos por validação cruzada por planta (seção 14).
   * Sem perfil (modelo minúsculo dos testes, demonstração): o quadro inteiro, com
     `anomaly_below` (0,30) e `low_confidence_below` (0,60).
3. **Retenção** (seções 16 e 17): NORMAL → imagem reduzida (320 px, JPEG 60) ou só
   metadados; PRAGA/DOENÇA/ANOMALIA → original + sequência do buffer; baixa confiança →
   original (nunca rebaixa a categoria). `normal_sample_every` guarda 1 a cada N normais em
   original para validação.
4. **Gravação**: o arquivo vai ao disco antes (`.part` + `fsync` + `rename`); depois, uma
   transação só grava tudo o que o quadro produziu. Queda entre os dois deixa um arquivo
   órfão, que o boot apaga. Nunca fica linha sem arquivo.
5. **Sincronização** no mesmo laço, limitada por passo (`sync_requests_per_step`), ou numa
   thread (`RoverCore.sync_forever`).

### Metadados de cada resultado (seção 16)

`id`, `device_id`, `mission_id`, `field_id`, `occurred_at`, `frame` (câmera, índice,
dimensões), `position` (lat, lon, altitude, fix, HDOP, rumo), `region` (planta/região, por
um `region_resolver` plugável — sem mapa de plantas no robô ainda, vem nulo),
`navigation` (linha de cada lado e metro ao longo dela, se a navegação publica — seção
14), `tiles` (com perfil: caixas, altura no chão e zona de cada recorte, distância,
altura e lado da câmera, campo de visão e as **probabilidades de cada recorte** — o quadro
normal sobe reduzido e a nuvem funde por planta a partir delas, seção 15), `imu`,
`sensors`, `model` (chave, versão, **SHA-256**, original/otimizado, **versão do dNATY**,
hardware-alvo), `hardware` (versão, acelerador, *providers*, **`simulated`**), `result`
(categoria, rótulo, confiança, top-3, motivo, latência), `retention` (nível, validade,
prioridade), `event_id`, `files` (papel, hash, tamanho), versões do Runtime e da Edge AI.
O resultado **é** o `AI_RESULT_CREATED` da seção 40 (um registro `kind=result`).

### Buffer circular (seção 15)

`buffer.pre_frames`, `post_frames`, `max_duration_s`, `max_side` (resolução do contexto),
`jpeg_quality`, `priority`. Por câmera; quadro mais distante do evento que
`max_duration_s` não entra; eventos sobrepostos compartilham quadros; o fim da missão fecha
as sequências abertas. Os quadros de contexto são gravados com papel `contexto-3` …
`contexto+3`, ligados ao ID do evento.

### Volume (seção 13) — medido em teste

`tests/test_retention.py`, com bytes reais (JPEG 1280x960 com textura, reduzidas e
contexto pela mesma função do Runtime, registros de uma missão real do Runtime):

| 10.000 quadros, 200 eventos (2 %) | Volume |
| --- | --- |
| Enviar tudo em resolução máxima | **6,04 GB** |
| Com a política padrão (NORMAL reduzida + eventos originais + 6 quadros de contexto) | **565 MB (9,4 %)** |
| Com NORMAL só em metadados | **385 MB (6,4 %)** |

Pipeline inteiro (200 quadros 640x480, 4 eventos, Edge AI de verdade sobre o modelo
minúsculo): 18,9 MB de originais → **1,89 MB** guardados (10,0 %).

## 4. Offline-first e reinício (seção 12)

- Sem rede, nada muda no laço: captura, IA, resultados, posição e eventos vão para o banco
  local; a fila cresce; a telemetria acumula.
- O Model Manager guarda a última designação do registry: offline, o robô usa o modelo que
  já tem **depois de reconferir o hash do arquivo**.
- Reinício no meio da missão: o boot apaga `.part`/órfãos, devolve à fila o que estava
  "ENVIANDO" (o progresso por chunk está gravado), recarrega o modelo e emite
  `MISSION_RESUMED`; o Core continua do último quadro gravado. Teste:
  `test_reinicio_no_meio_da_missao_continua_sem_perder_nem_duplicar` e, contra o SaaS,
  `saas/tests/test_rover_integracao.py`.
- Disco cheio: a retenção apaga primeiro o que já subiu e venceu, depois o que já subiu,
  depois as reduzidas NORMAL que não subiram (o resultado continua e sobe sem imagem).
  **Original de evento que não subiu nunca é apagado.** Gera `STORAGE_PRESSURE`.

## 5. Modelos no Rover (seções 9 e 22)

1. `GET /v1/device/models` (SaaS) diz, por talhão, a versão em PRODUÇÃO do registry com o
   SHA-256 do artefato (o otimizado pelo dNATY quando houver).
2. O artefato chega por um `ArtifactSource` — hoje `FolderArtifactSource` (pasta com
   `<sha256>.onnx` + `.labels.json`; em campo, a área de *staging* do OTA ou um pendrive de
   serviço). Nada é confiável por estar na pasta.
3. Antes de carregar, o SHA-256 **do arquivo** é comparado ao do registry. Diferente →
   `MODEL_REJECTED` (crítico) e não carrega. Designação sem hash → não carrega. Arquivo sem
   designação → nunca carrega. **Missão não começa sem modelo do registry.**
4. As versões (runtime, Edge AI, modelo, dNATY, hardware) vão em `POST /v1/device/session`
   quando mudam — é daí que a frota mostra conforme/divergente/fora do registry.

## 6. Missão

`RoverCore.accept_cloud_mission()` busca `GET /v1/device/missions/current` e só aceita
depois das cinco conferências do protocolo (assinatura Ed25519 com a chave pública da
plataforma, destinatário, validade, sequência e nonce — `mission.verify_command`). Para
operação sem nuvem e para o simulador, `start_mission(mission_id, field_id)`.

## 7. Plugar hardware real (adapters)

O Runtime só conhece as interfaces de `hal/interfaces.py`. Para o hardware definitivo,
escreve-se um adapter por dispositivo e monta-se o `Hardware` no programa de
inicialização do robô — nada acima da HAL muda.

```python
from valtexa_rover.hal.interfaces import DeviceInfo, Frame, HardwareUnavailableError

class CameraGigE:                       # exemplo: câmera industrial
    def __init__(self, camera_id: str, serial: str, clock): ...
    def capture(self) -> Frame:
        # dispara, recebe o buffer, codifica em JPEG (qualidade alta) e devolve Frame
        # com captured_at em UTC. Timeout/cabo solto → raise HardwareUnavailableError.
        ...
    def describe(self) -> DeviceInfo:
        return DeviceInfo("camera", "GigE <modelo>", simulated=False, detail={...})

hardware = Hardware(
    cameras=[CameraGigE("cam-0", "...", clock), ...],
    gps=GnssRtk("/dev/ttyACM0"),        # NMEA/UBX → GpsFix (fix: none/gps/float/fixed)
    imu=ImuSpi(...), sensors=SensoresCan(...),
    storage=LocalDiskStorage("/var/lib/valtexa-rover", quota_bytes=...),
    network=HttpNetwork("https://api.valtexa..."),
    accelerator=OnnxCpuAccelerator(prefer=["TensorrtExecutionProvider",
                                           "CUDAExecutionProvider", "CPUExecutionProvider"],
                                   name="Jetson Orin"),
    controller=ControladorCan(...), hardware_version="Rover Edge V1",
)
core = RoverCore(hardware, "/var/lib/valtexa-rover", DeviceIdentity.load(".../identity.json"),
                 config=RuntimeConfig.load("/etc/valtexa/rover.json"),
                 artifact_source=FolderArtifactSource("/var/lib/valtexa-rover/ota"))
core.boot()
```

Contrato de todo adapter: falha é `HardwareUnavailableError` (nunca `None` silencioso),
horário em UTC, coordenadas WGS84, `describe()` honesto (`simulated=False` só para
hardware de verdade). Para um acelerador novo (NPU), basta o *provider* do ONNX Runtime do
fabricante aparecer em `providers()`.

A identidade (`identity.json`, 0600) nasce no provisionamento: o robô gera o par Ed25519
(`runtime.client.generate_keypair`) e manda só a pública para `POST /v1/device/provision`.
A chave privada nunca aparece em log (`repr` a esconde).

## 8. Configuração (`rover.json`)

`valtexa-rover config-padrao` imprime tudo com os padrões; `valtexa-rover validar-config`
confere. Exemplo — NORMAL só em metadados, buffer maior, chunk de 1 MB:

```json
{
  "retencao": {"rules": {"NORMAL": {"tier": "METADADOS", "retention_days": 3}},
               "normal_sample_every": 200},
  "buffer": {"pre_frames": 5, "post_frames": 5, "max_side": 1280},
  "sync": {"chunk_size": 1048576}
}
```

## 9. Troubleshooting

- `valtexa-rover status --dados <pasta>` — missão, modelo ativo, fila por status e
  prioridade, telemetria não enviada, se a sincronização está bloqueada.
- `sync_blocked: true` — a nuvem respondeu `403` (robô suspenso, credencial revogada). O
  Runtime **para de insistir** (protocolo, seção 7); resolver na plataforma e limpar a
  chave `sync_blocked` do banco local.
- Itens `FALHOU` na fila: `last_error` diz o motivo (HTTP 4xx definitivo, conflito de
  hash, arquivo removido pela retenção).
- Logs em `<dados>/logs/rover.log` (JSON por linha).

## 10. OTA (Parte 5)

Ver `docs/OTA.md`. Resumo: `GET /v1/device/ota` → envelope assinado com a **chave OTA**
(vem no provisionamento, `identity.ota_public_key`) → download em pedaços com retomada →
hash do pacote e de cada arquivo → instalação lado a lado em `ota/slots/` → saúde →
ponteiro `ota/current.json` trocado de forma atômica → runtime/Edge AI/configuração ficam
em teste até o próximo boot confirmar; o lançador volta sozinho se não confirmar. Nada é
instalado com missão em execução. Modelo: vai para a pasta de artefatos do Model Manager
— quem decide qual roda continua sendo o registry.

## 11. Simulador (Parte 5, seção 41)

`valtexa_rover.sim` monta o **mesmo** `RoverCore` (Edge AI, missão, sync, OTA) sobre a HAL
simulada. Não existe um "robô de mentira" separado: o que muda é só o adapter.

```bash
# contra uma plataforma no ar (protocolo real, HTTPS):
python -m valtexa_rover.sim --url http://127.0.0.1:8000 --token vtx_prov_... \
    --dados /tmp/rover-sim --passos 120 --sem-rede 30-60
# o ponta a ponta inteiro, com a plataforma subindo sozinha:
cd ../saas && python -m simulator --rover-runtime
```

- `CampoSimulado.gerar(lon, lat, linhas, plantas_por_linha, espacamento_linhas_m,
  espacamento_plantas_m, doentes={...})` — o talhão e o "chão de verdade";
- `CameraDeCena` — fotografa a planta mais próxima do GPS (verde/vermelho/azul com ruído);
  opcionalmente anexa `VTXSIM:<rótulo>` ao fim do JPEG para o analisador **simulado** da
  nuvem (a Edge AI não lê a marca);
- `PlanoDeRede(sem_rede=[(início, fim)])`, `RoverSimulado.reiniciar()`, câmera
  indisponível por janela, `ControladorDeRota` (diz quando a rota acabou);
- `region_resolver` = o mapa de plantas do campo: cada resultado diz a planta (`region`);
- o modelo é `sim/modelo_demo.onnx` (282 bytes, SHA-256 `f29e7554…`), **não agronômico**.

O simulador antigo do `saas/simulator` (`python -m simulator` sem opção) continua como
**legado**: ele encena drone, dock e cópia 3D com um cliente próprio do protocolo
(`simulator/client.py`), que não é o `rover/`. O caminho do Rover é o `--rover-runtime`.

## 12. Implantar (Parte 5, F19)

`valtexa-rover servico --dados ... --identidade ... --config ... --hal modulo:funcao
--url https://...` é o Runtime em serviço: boot (recupera disco, retoma missão, confirma
OTA), pergunta por missão assinada a cada `--busca-missao` s, laço de captura, conclui
quando o controlador diz `route_done`, informa `ack`/`complete` à plataforma, OTA fora de
missão, SIGTERM fecha o quadro em curso. `--hal` é a função do integrador que monta a
`Hardware` real (`funcao(dados, url, clock) -> Hardware`); para homologação sem hardware,
`valtexa_rover.sim:hal_de_demonstracao` (SIMULADO).

Arquivos em `rover/deploy/`: `valtexa-rover.service` (systemd, com o lançador A/B do OTA,
`ProtectSystem=strict`, escrita só em `/var/lib/valtexa-rover` e `/etc/valtexa`),
`rover.json.exemplo` (todos os padrões, validado), `rover.env.exemplo` (URL e HAL; sem
segredo). Instale o pacote **sem** `-e`: o `PYTHONPATH` do slot do OTA precisa vencer o
`site-packages`.

## 13. Estresse (Parte 5, seção 42)

`python -m valtexa_rover.sim.estresse --passos N [--cota-mb M]` — missão longa com rede
intermitente, rajadas de 503/429, reinícios, câmera fora, queda no meio de arquivo e disco
pequeno; confere no fim que cada registro chegou **uma vez**, que todo original de evento
chegou, que a missão contou cada quadro uma vez e que a fila zerou. Números em
`docs/ESTRESSE.md`.

## 14. Rover V3 (Jetson Orin NX) — instalar no hardware

Adapters em `valtexa_rover/hal/rover_v3.py`, montados pelo `hardware.json`
(`deploy/hardware.rover-v3.json.exemplo`):

| Peça (BOM rev E) | Adapter | Estado |
|---|---|---|
| Jetson Orin NX 16 GB | `JetsonAccelerator` (TensorRT → CUDA → CPU), `JetsonSystemMonitor` (`tegrastats`) | testado com saída real do `tegrastats`; providers só na Jetson |
| 6 × OAK-D Pro W PoE (laterais) | `OakDCamera` (`depthai` 2.x, foto parada 12 MP + profundidade alinhada) | **não rodou numa OAK-D**; SDK falso nos testes |
| u-blox ZED-F9P | `NmeaGnss` (GGA/RMC, RTK fixo/flutuante, fix velho recusado) | testado com sentenças NMEA reais |
| Xsens MTi-630 (CAN) | `XsensCanImu` (`python-can`, IDs e escalas no `hardware.json`) | decodificação testada; **IDs a conferir no MT Manager** |
| Livox Mid-360 + navegação | `NavegacaoPorArquivo` (`/run/valtexa`) | testado com o laço do serviço |
| MCU de segurança + motores | — | protocolo não definido; o Runtime não comanda movimento |

**Instalar** (JetPack 6, Python 3.10 — o pacote roda em 3.10+; conferido em 3.10.20):

```
sudo deploy/jetson/instalar.sh --ort-wheel onnxruntime_gpu-<versão>-cp310-cp310-linux_aarch64.whl
```

O ONNX Runtime com CUDA/TensorRT para Jetson vem da NVIDIA (Jetson AI Lab / Jetson Zoo),
não do PyPI: o instalador põe as dependências uma a uma e o pacote com `--no-deps`, para o
`onnxruntime` de CPU do PyPI não sobrescrever o da NVIDIA (um teste confere a lista contra o
`pyproject.toml`). Ele cria o usuário `valtexa`, o venv em `/opt/valtexa/venv`, a
configuração em `/etc/valtexa` (sem sobrescrever a existente), a unidade systemd com o
complemento `rover-v3.conf` (grupos `video render dialout`, `/run/valtexa`, cache do
TensorRT), o `can0` pelo systemd-networkd (500 kbit/s) e o nome fixo `/dev/valtexa-gnss`
para o ZED-F9P (regra udev). Com TensorRT, o último lote de recortes de cada quadro é
completado até `tile_batch`: uma forma só, um motor compilado só.

**Primeira montagem**, nesta ordem:

1. `valtexa-rover diagnostico --dados /var/lib/valtexa-rover --salvar /tmp/diag` — lê cada
   peça uma vez (câmeras com distância medida, GNSS, IMU, acelerador, disco, navegação) e
   diz qual falhou, sem identidade nem rede. Os quadros salvos servem para conferir
   foco, exposição e o lado de cada câmera.
2. `valtexa-rover provisionar --url ... --serial ... --identidade .../identity.json` — a
   chave Ed25519 nasce no robô; o token de uso único vem pela entrada padrão (ou
   `VALTEXA_TOKEN_PROVISIONAMENTO`), nunca na linha de comando; só por HTTPS; não
   sobrescreve uma identidade existente.
3. `valtexa-rover preparar-modelo banana_foliar.onnx --dados /var/lib/valtexa-rover` —
   carrega o modelo e o perfil `.edge.json` **neste** acelerador, mede um quadro sintético
   de 12 MP e copia `<sha256>.onnx` + rótulos + categorias + perfil para a área de staging
   do OTA. Sem perfil, recusa (`--sem-perfil` para forçar). Designe na plataforma o SHA-256
   impresso para o talhão.
4. `systemctl enable --now valtexa-rover`.

**Navegação.** A pilha de navegação (fora deste pacote) e o Runtime trocam três arquivos
JSON em `/run/valtexa` (o diretório é do serviço; a navegação entra no grupo `valtexa`):
`missao.json` (Runtime → navegação: a missão assinada com a rota, ou `{"id": null}`),
`navegacao.json` (navegação → Runtime: `{"mission_id", "route_done", "row_left",
"row_right", "x_m"}`) e `parar.json` (pedido de parada com motivo). `route_done` de outra
missão é ignorado; `row_left`/`row_right`/`x_m` vão em cada resultado e dão à fusão por
planta a linha e a posição exatas. Sem a navegação, a missão não termina sozinha — o
operador pode escrever `{"route_done": true}` no `navegacao.json`.

**Café (a cultura do produto).** O perfil do café pede a **cabeça sobre a torre de visão**
(bloco `head` do `.edge.json`): cada recorte passa pela torre do SigLIP 2 em ONNX
(`<sha>.encoder.onnx`, 355 MB, conferida pelo SHA-256) e pela cabeça
(`<sha>.head.npz`); a CNN sozinha acerta 6,5% no café. Para preparar:
`valtexa-rover preparar-modelo coffee_foliar.onnx --encoder framing_vision.onnx --dados ...`
— copia os seis arquivos para o staging (modelo, rótulos, categorias, perfil, cabeça e
torre). Medido em CPU de PC: ~9,8 s por quadro (70 recortes); na Jetson, a torre vai para
a GPU e o tempo precisa ser medido na bancada antes de definir a cadência das câmeras.
Limiares do café: provisórios (calibrados em mosaico de folhas; ver `MODELS.md` de
VALTEXA-B, "Café no Rover") — a primeira coleta no Sul de Minas recalibra.

**Modelos reais e limiares medidos.** `banana_foliar.onnx` e `citrus_foliar.onnx`
(MobileNetV2, 9,5 MB, entrada 224 × 224) com o perfil calibrado por `training.fit_edge`
em VALTEXA-B: plantas de três fotos (uma por andar de câmera), validação cruzada por
planta, orçamento de 5% de quadros sadios virando evento. Números em `MODELS.md` de
VALTEXA-B (seção "Edge AI do Rover"). Limite: fotos de folha agrupadas, não o Rover no
pomar — a validação de campo deve remedir com o registro do Rover.

**Correção da lente.** A `OakDCamera` lê a calibração de fábrica de cada câmera
(`readCalibration()` do depthai: Brown-Conrady ou fisheye Kannala-Brandt) e desfaz a
distorção antes de recortar (`hal/lente.py`; `cv2.remap` com o OpenCV do extra `rover-v3`,
NumPy sem ele). A foto sai pinhole com o mesmo fx, e o campo de visão efetivo
(`2·atan(W/2fx)`) vai em cada quadro (`hfov_deg`) para a IA embarcada e para a nuvem — o
tamanho do recorte em mm usa esse número, não o do perfil. `hardware.json`:
`"corrigir_lente": false` desliga; `"resolucao"` escolhe o modo do sensor. O diagnóstico
mostra, por câmera, "corrigida" ou "sem correção" (calibração ilegível).

**O que falta conferir no hardware** (não dá para testar sem ele): a fila de foto parada
e o alinhamento da profundidade da OAK-D; os IDs/escalas CAN da Xsens instalada; o
pinmux do CAN da placa-base (`mttcan`); a latência real por quadro na Orin com TensorRT
(medida aqui só em CPU de PC); o lado e a altura de cada câmera no `hardware.json`.

## 15. Fusão por planta na nuvem

Cada resultado com `tiles` vai, pelo worker do SaaS (`services/fusao_borda.py`, a cada
30 s), para `POST /fields/{id}/frames` de VALTEXA-B: as probabilidades de todos os
recortes entram na fusão por planta com a calibração medida para recortes da borda
(`<pesos>.edge-fusion.json`, `training.fit_edge fusao` — sem TTA e sem o portão de
enquadramento, que só a nuvem tem). Por isso a incidência e a prescrição contam também
as plantas **sadias**, não só os eventos que subiram em original.

A linha e o metro de cada quadro vêm da navegação (`navigation`) ou, sem ela, do GNSS
projetado no talhão (`row_azimuth_deg`, `row_spacing_m`; linhas `G1`, `G2`, ... a partir
do lado esquerdo do azimute, supondo o Rover no meio da rua). Quadro sem geometria fica
`sem_geometria` com o motivo — nada inventa posição. Cada resultado vai uma vez
(`edge_record.fusion_status`; a IA recusa o mesmo `frame_id`), e a passada da missão só é
fechada depois que os resultados chegaram (o `frames_done` do `MISSION_COMPLETED`, ou 6 h
sem resultado novo). O relatório da missão traz `calibrated_fusion` (incidência da IA e
quantos resultados entraram). Rotas: `GET /v1/fields/{id}/plants/incidence` e
`POST /v1/fields/{id}/plants/prescription` (recomendação planta a planta; recurso
`prescricao` do plano; nada é aplicado sem aprovação).

## 16. Centro de comando

Console → **Plataforma → Centro de comando** (`/plataforma/comando/<rover>`), permissão
`platform:rover:control` (só operações da plataforma). Uma tela por robô: modo, PARAR,
controle remoto, teste de cada peça, foto de cada câmera, modelo de IA, configuração,
telemetria das últimas 2 h e registro dos comandos. Rotas em `api/routers/platform_rover.py`:

| Rota | O quê |
|---|---|
| `GET /v1/platform/rovers/{id}/painel` | tudo do robô numa resposta (o console busca a cada 2 s) |
| `POST /v1/platform/rovers/{id}/comandos` | `diagnostico`, `foto`, `modo`, `parar`, `config`, `recarregar_modelos`, `reiniciar` |
| `POST .../comandos/{cid}/cancelar` | cancela um pendente |
| `POST .../teleop`, `POST .../teleop/parar` | controle remoto (só em modo `manual`; 409 fora dele) |
| `GET .../modelos`, `PUT .../modelo` | fixa um modelo no robô (ou solta) |
| `GET /v1/device/commands`, `POST /v1/device/commands/{cid}/result`, `GET /v1/device/teleop` | lado do robô |

**Assinatura.** Cada comando sai assinado como a missão (envelope Ed25519, validade de
2 min, sequência própria por robô); o Runtime (`runtime/comandos.py`) confere assinatura,
prazo e sequência (`last_remote_command_sequence`, separada da das missões) antes de
executar, e devolve o resultado. Comando forjado ou repetido é recusado e registrado.

**Controle remoto.** O console manda o ponto de ajuste a cada 250 ms; cada um vale 1 s
(homem-morto) e é limitado a 0,5 m/s e 1,0 rad/s. O Runtime busca `/device/teleop` a cada
volta do laço e escreve `/run/valtexa/teleop.json` (`{"v", "w", "valid_until", "seq"}`);
sem ponto válido (204 ou rede caída), escreve zero. **A navegação tem de parar sozinha
quando `valid_until` passa** — o Runtime não comanda motor.

**Modo.** `modo` escreve `/run/valtexa/controle.json` (`{"modo": "autonomo"|"manual"|
"pausado"|"parado", "at"}`); `parar` também escreve `parar.json` com o motivo. Quem move o robô é a
pilha de navegação e o MCU de segurança, que precisam ler esses arquivos (protocolo do MCU
ainda não definido).

**Configuração.** `config` mescla o JSON no `rover.json`, valida, grava de forma atômica
(guarda `rover.json.anterior`) e reinicia o serviço (código de saída 75; o systemd sobe de
novo). Predefinições no console: **coleta** (`retencao.normal_sample_every = 1`, guarda
toda foto em original — para validação e treino) e **normal**.

**Modelo fixado.** Um modelo fixado no robô vale para os talhões da mesma cultura, acima da
designação do talhão; soltar volta à designação. `recarregar_modelos` força a busca.

## 17. Validação de campo (rótulo do agrônomo → recalibração e treino)

1. Missão em modo coleta sobre fileiras de pés numerados.
2. O agrônomo anota cada pé (`PUT /v1/fields/{id}/validacao/{missão}/{pé}`: `sadia`,
   `ferrugem`, ... com a incidência em %; permissão de revisão de diagnóstico).
3. `GET .../validacao/{missão}/relatorio`: robô contra agrônomo (positivo do robô =
   `suspeita`/`doente`/`revisar`), com intervalo de Wilson.
4. `GET /v1/fields/{id}/validacao.csv` → `rotulos.csv` (`field_id,plant_id,label,pass_id`).
5. Recalibrar em VALTEXA-B com o registro do campo (`FIELDS_TILE_LOG=true`):
   `python -m training.fit_edge fit --model coffee-foliar --frames <registro>.jsonl --labels rotulos.csv`
   e `python -m training.fit_fusion fit --model coffee-foliar --frames <registro>.jsonl --labels rotulos.csv --fields-dir fields`.
6. Treino: `python scripts/exporta_originais.py --mission <missão> --dest originais/`
   (só registros com consentimento de treino) e
   `python -m training.import_campo --labels rotulos.csv --registro <registro>.jsonl --originais originais/ --out data/coffee-campo`
   — recortes de pés sadios entram como `Healthy`; os de pés doentes vão para `_revisar`
   (nem todo recorte de um pé doente tem lesão). O `campo.csv` diz a planta de cada recorte.
7. Aprendizado contínuo da cabeça com esses recortes (seção 24):
   `python -m training.adapt_campo adapta --model coffee-foliar --campo data/coffee-campo --escopo <fazenda>`
   e, se o portão aprovar, `promove`.

Guia de validação do robô inteiro (fases, critérios, o que comprar):
https://claude.ai/artifact/KiKdRqL7X6i9Cf73s4TdDn

## 18. Ensaios automáticos (Parte 11, rodada 3)

Pacote `valtexa_rover/ensaios/` — guia completo, critérios e portões em
`VALTEXA-SAAS/docs/plataforma/VALIDACAO_ROVER.md`.

| Arquivo | O quê |
|---|---|
| `ensaios/base.py` | `Param` (tipo, faixa, obrigatório), `Contexto` (hardware, relógio, fontes, progresso, cancelamento), `rodar()` → laudo padrão (`status`, `criterios`, `medidas`, `amostras`, `simulado`) |
| `ensaios/{calor,cameras,profundidade,gnss,imu,seguranca,rede,energia}.py` | um ensaio por peça; cada um diz em `hardware_real` o que falta conferir no robô |
| `ensaios/estatistica.py` | percentis, CEP50/CEP95/2DRMS, inclinação pela gravidade — refeitas na plataforma |
| `ensaios/__init__.py` | `CATALOGO`, `fontes_do_runtime(core)` (carga = modelo ativo, térmica = tegrastats, rede = canal assinado, energia = sensores) e `ExecutorDeEnsaios` (thread, um por vez) |
| `ensaios/simulado.py` | bancada **SIMULADA** com injeção de falha (câmera que perde/pula/atrasa, profundidade com viés, GNSS lento/ruidoso/com perdas, IMU com eixos trocados, navegação lenta/surda, Jetson que esquenta, rede que perde) e `RelogioDeBancada` |
| `ensaios/fase0.py` | a FASE 0 em ordem e o laudo em Markdown |

**Comando `ensaio`** (`runtime/comandos.py`): o mesmo canal assinado do centro de comando
(sequência própria, repetido/forjado recusado). Não executa na hora: inicia a thread,
manda `POST /v1/device/commands/<id>/progresso` (no máximo a cada 2 s) e, no fim, o laudo
pelo `result` de sempre — sem rede, o laudo fica guardado e vai na próxima busca.
Recusado com missão em curso ou outro ensaio rodando; `config` e `reiniciar` também
esperam o ensaio acabar, e o serviço não aceita missão com ensaio em curso. `ensaio_cancelar` `{"command_id"}` para no próximo ponto.

**HAL** (mudanças compatíveis): `ImuReading.gyro_dps` opcional (Xsens com `gyro_id` e
`gyro_escala_dps` no `hardware.json`, a conferir no MT Manager);
`parse_tegrastats` devolve também `cpu_freq_mhz`, `gpu_freq_mhz` e `temperaturas_c`
(`JetsonSystemMonitor.ultima_tegrastats()`); `NavegacaoPorArquivo.stop()` devolve o pedido
(`{"reason", "at"}`) — a navegação confirma com `parado_em` + `parada_ref` no
`navegacao.json`; o Runtime zera o `teleop.json` quando o ponto vence sem outro.

**Rodar sem plataforma**: `python -m valtexa_rover.sim fase0 --saida laudo.md` (bancada
simulada). **Com a plataforma**: `cd saas && python -m simulator --fase0`.

## 19. Comunicação da fazenda: rede "via base" (Parte 12, rodada 3) — ARQUIVADO

> **ARQUIVADO em 20/10/2026.** Não existe mais base da fazenda nem rádio: cada Rover tem a
> própria Starlink e fala com a plataforma por HTTPS direto. O código está em
> `archive/legacy/radio-fazenda/` (com o README de recuperação) e o projeto completo da
> comunicação pela base em `VALTEXA-BASE/docs/COMUNICACAO_FAZENDA.md`.

## 20. Enlace à prova de falha (Parte 13, rodada 3) — ARQUIVADO

> **ARQUIVADO em 20/10/2026.** Não existe mais base da fazenda nem rádio: cada Rover tem a
> própria Starlink e fala com a plataforma por HTTPS direto. O código está em
> `archive/legacy/radio-fazenda/` (com o README de recuperação) e o projeto completo da
> comunicação pela base em `VALTEXA-BASE/docs/COMUNICACAO_FAZENDA.md`.

## 21. Estação de pilotagem e vídeo ao vivo

Na estação de pilotagem do centro de comando, o vídeo das câmeras, o **controle de
videogame**, o teclado e o joystick da tela ficam no mesmo lugar, com o painel de voo (HUD)
sobre a imagem. O HUD mostra:

* modo e fonte do comando;
* velocidade pedida e enviada, e marcha;
* bateria e RTK;
* idade do quadro e quadros por segundo;
* o caminho nuvem → base → rádio.

**Controle de videogame** (Gamepad API do navegador, mapeamento "standard"):

| Comando | Função |
|---|---|
| Analógico esquerdo | Acelera |
| Analógico direito | Vira |
| **LB ou RB segurado** | Homem-morto: soltou, o ponto vai a zero |
| **B** | **PARAR** (comando assinado) |
| ↑ / ↓ | Marcha |
| X / Y | Trocam a câmera |

O analógico tem zona morta com reescala e curva fina perto do centro
(`web/src/lib/pilotagem.ts`).

**Teclado:** W A S D ou setas (segurar a tecla é o homem-morto), espaço = PARAR, Q/E =
marcha.

Qualquer que seja a fonte, cada ponto vale 1 s no servidor e só vale em modo Manual.

**Vídeo, de ponta a ponta:**

1. A estação pede as câmeras, a taxa e o tamanho: `POST /v1/platform/rovers/{id}/video`.
   O pedido vence em 12 s; a tela renova a cada 5 s. Começar a assistir fica na auditoria
   do cliente.
2. O Runtime fica sabendo pelo cabeçalho `X-Valtexa-Video` da consulta de comandos que
   ele já faz, sem consulta a mais.
3. O `TransmissorDeVideo` (`runtime/video.py`) manda `POST /v1/device/video/{camera}`
   (JPEG assinado):
   * **em missão**, usa o último quadro que o laço já capturou e não chama a câmera;
   * **fora de missão**, usa a prévia da OAK-D (`capture_preview`, saída de vídeo separada
     da foto de 12 MP);
   * **com ensaio em curso**, não captura.
4. A **qualidade é adaptativa**: quadro que demora mais que o intervalo faz o próximo sair
   menor (de 240 px a 1280 px). O laço nunca morre: um erro vira `ultimo_erro` e segue.
5. Pela base da fazenda, cada câmera é um slot "só o último vale" em P3, com prazo de 3 s:
   quadro velho cai no caminho, nunca passa na frente de comando ou teleop, nunca
   enfileira.
6. A plataforma guarda **só o último quadro** de cada câmera por 15 s, na coordenação
   compartilhada (Redis com várias réplicas). A estação busca o último e mostra a idade.

**Contrato entre os sistemas** (`saas/tests/test_contrato_sistemas.py`): o teste lê o
código do Rover e da base e confere que toda rota que eles chamam existe no OpenAPI da
plataforma com o mesmo método. Confere também os cabeçalhos combinados e a prioridade de
cada rota no enlace.

**Provas:**

* `rover/tests/test_video.py` (10 testes, inclusive a prévia da OAK-D com SDK falso);
* `saas/tests/test_video_ao_vivo.py` (5);
* `saas/tests/test_base_fazenda.py`: vídeo pela base, rádio com 15% de perda e Rover em
  missão;
* `web/src/lib/pilotagem.test.ts` (7);
* roteiro de navegador com **controle de videogame simulado** (Edge):
  * vídeo a 222 ms de idade e 4,3 quadros/s;
  * sem LB/RB o Rover não anda; com RB, `teleop` ativo no servidor; soltou, parou;
  * B mandou PARAR e o Rover confirmou "parado".

**Não testado em hardware:** a prévia na OAK-D real e o vídeo pelos rádios reais. A banda
do sub-GHz limita a qualidade: a adaptação desce até 240 px.

## 22. Modelo pronto no boot

**Pré-carga do modelo** (`runtime/core.py`, `precarrega_modelo`):

* no boot sem missão, o Runtime já carrega o modelo do último talhão (`last_field_id`) ou
  o designado ao robô. A primeira missão não espera e o painel não mostra
  "Erro: model_missing" com o robô parado;
* `model_missing` só vira falha quando uma missão roda sem modelo, ou quando um modelo
  designado não carrega (`modelo_indisponivel`);
* "já carregado" confere também o hash do perfil (`.edge.json`) e da cabeça (`.head.npz`).
  Um perfil recalibrado com o mesmo ONNX agora recarrega; antes ficava o velho na memória;
* o comando `recarregar_modelos` com o robô parado já deixa o modelo novo pronto.

(O mapa de cobertura do rádio na missão foi arquivado com a base em 20/10/2026.)

**Provas:** `tests/test_precarga.py` (4) e `tests/test_model_manager.py`.

## 23. Nível 2: percepção ativa e fusão RGB + profundidade

O Rover deixa de decidir com uma foto só, do jeito que ela vier. Duas peças, no pacote do
robô (`VALTEXA-ROVER`); a terceira — aprender com o campo — fica na IA (seção 24).

**Fusão RGB + profundidade** (`edge_ai/rgbd.py`, seção `profundidade` do `rover.json`):

* a OAK-D entrega o mapa de profundidade alinhado à foto, reduzido (lado maior 320 px) e
  com a mesma correção de lente (`Undistorter.apply_depth`, vizinho mais próximo — média
  entre folha e fundo é uma profundidade que não existe). Vai em `Frame.depth`;
* a distância da planta sai do **pico mais próximo** do histograma no centro do quadro, não
  da mediana: pelos vãos do dossel aparece a rua a 2–3 m, e a mediana puxa para lá — a
  escala dos recortes de 150 mm erraria;
* recorte medido e fora da faixa da planta (`frente_mm`/`tras_mm` em volta do pico) é
  **fundo**: com `modo = filtrar` não vai ao modelo nem à evidência. Recorte sem medida
  suficiente fica. Cobertura abaixo de `min_cobertura_quadro` (sol, superfície lisa): a
  profundidade não decide nada;
* **abertura do dossel**: a fração dos pixels medidos atrás da planta, no quadro e por
  terço. Indicador **não calibrado** de desfolha (ferrugem derruba folha); vai no resultado
  (`rgbd.canopy_openness`, `canopy_openness_by_zone`) para ser comparado com o agrônomo;
* `modo = anotar` só mede e manda; `desligado` ignora o mapa.

**Percepção ativa** (`runtime/percepcao.py`, seção `percepcao_ativa`):

1. *dúvida*: evidência a menos de uma margem do limiar de evento (padrão: a distância entre
   quase-evento e evento que o `fit_edge` mediu — 0,036 no café), quase-evento da regra por
   contagem, anomalia, ou profundidade insuficiente. A região que decidiu são os recortes a
   favor da doença líder, com um recorte de folga;
2. *olhar de novo*: até `max_vistas_extras` (2). Na OAK-D, `capture_region` mede a
   exposição (e o foco, com `autofoco`) **naquela região** e volta ao quadro inteiro;
3. *alinhar*: correlação de fase entre as fotos reduzidas (decodificação parcial do JPEG).
   Vale se o pico passa de `min_registro` **e** o deslocamento cabe no movimento do Rover
   (quase todo horizontal: `max_desvio_vertical`) — um reflexo forte numa foto só puxa a
   correlação para um deslocamento absurdo, e aí a vista não casa recorte a recorte;
4. *fundir*: a vista extra só roda o modelo nos recortes da região deslocada (no café, ~10
   em vez de ~70, a 131 ms cada na CPU). Cada recorte da principal vira a média das vistas
   que o cobriram e a decisão sai pela **mesma regra calibrada** (`decide_tiles`), com o
   mesmo número de recortes — a escala do escore não muda, nada é contado duas vezes. Para
   quando a dúvida se resolve;
5. *o que continua incerto* sobe em ORIGINAL com as vistas (`vista1`, `vista2`) e
   `retention.for_validation` — o exemplo que o agrônomo precisa ver e o que mais ensina ao
   aprendizado contínuo. Profundidade ruim **sozinha** não sobe original: é do sensor, e
   uma OAK-D sem medida encheria o disco numa missão;
6. *orçamento*: `max_extras_por_minuto` por câmera e `tempo_max_ms` de inferência extra
   estimada. Fora dele, o bloco diz `skipped`.

Com profundidade insuficiente, a vista extra é do quadro inteiro; se ela vier com medida,
**substitui** a principal (`promoted_view`). Os `tiles` que sobem para a fusão por planta
continuam os da vista principal, crus (a fusão da nuvem foi calibrada com uma vista por
quadro); o que as vistas disseram vai no bloco `percepcao` do resultado (`reasons`,
`views[]` com deslocamento, pico, região e latência, `before`/`after`, `fused_tiles`,
`resolved`, `for_validation`). Modelo sem perfil de recortes (o de demonstração): a
percepção ativa não age.

**`hardware.json`, por câmera** (padrões entre parênteses): `profundidade_curta` (sim —
disparidade estendida e monos em 400P: a câmera fica a ~27 cm da saia, abaixo do mínimo do
estéreo sem isso, pela documentação da Luxonis), `autofoco` (não — a Pro W é de foco fixo
na maioria das versões), `espera_exposicao_s` (0,25), `corrigir_lente_profundidade` (sim).

**Pausa da navegação** (opcional): com `percepcao_ativa.pedir_pausa_s > 0` o Runtime
escreve `/run/valtexa/percepcao.json` (`pause_until`, `reason`). Só vale se a pilha de
navegação ler o arquivo — o protocolo não está definido, por isso o padrão é não pedir.

**Cabeça e perfil casados**: o `.edge.json` pode dizer `head.head_sha256`; o motor recusa
o perfil com outra cabeça ("recalibrado"). O `fit_edge` grava o SHA; o `adapt_campo
promove` carimba o perfil instalado (seção 24).

**Não rodou no hardware**: alinhamento da profundidade à foto (com e sem a correção de lente
no mapa), a região de exposição e a profundidade curta da OAK-D precisam da bancada antes do
campo. Roteiro: `valtexa-rover ensaio profundidade` contra a trena a 0,27 m; sobrepor o mapa
à foto numa folha recortada sobre fundo distante; numa linha de café, comparar
`excluded_tiles` e `canopy_openness` com o que se vê, e a taxa de `for_validation` por
missão (o orçamento padrão limita a 6 vistas extras por câmera por minuto).

**Provas:** `rover/tests/test_rgbd.py` (12: pico da planta, fundo fora da evidência — o
objeto vermelho na rua deixa de ser ferrugem —, mapa ruim não filtra, abertura do dossel,
lente sem interpolar) e `test_percepcao.py` (alinhamento, dúvida, reflexo que some, lesão
confirmada pela exposição na região, dúvida que vai para a validação com as vistas,
orçamento, troca da vista por profundidade, falha de câmera, pausa, OAK-D com SDK falso,
perfil de outra cabeça recusado, o Runtime gravando e enfileirando tudo).

## 24. Nível 2: aprendizado contínuo (VALTEXA-B)

O que o agrônomo valida volta para o modelo — sem que ele esqueça o que já sabia. Detalhe
em `VALTEXA-B/README.md` ("Aprendizado contínuo no campo") e a medição em `MODELS.md`.

* **o que aprende**: só a cabeça sobre o embedding do SigLIP 2 (no café, a resposta
  inteira); torre e CNN congeladas. Treino NumPy na CPU, segundos por rodada
  (`app/inference/continual.py`, `training/adapt_campo.py`);
* **sem esquecer**: replay (memória de embeddings por classe com cota por origem — o
  acervo público e cada rodada de campo), destilação dos logits gravados quando o exemplo
  entrou e âncora L2-SP nos pesos de partida; a época sai da validação que a cabeça nunca
  treinou, entre as que não pioram o que já se sabia;
* **portão de regressão**: conjuntos congelados (o teste do treino, o de cada rodada
  aprovada) e comparação pareada com bootstrap. Não-inferioridade no agregado (−2 pontos),
  nenhuma piora demonstrada por conjunto, por classe (até −5) e no alarme falso em sadia
  (até +2), e ganho no campo. `laudo.md` por rodada;
* **monitoramento**: `monitora` compara a cabeça em produção com a anterior a cada lote novo
  do agrônomo (sai 3 em regressão) e `reverte` volta — a cabeça e o perfil da borda medido
  com ela. Histórico também em `GET /models/{key}/continuo`;
* **o elo com o Rover**: o `promove` carimba `head.head_sha256` no `.edge.json`. Se a
  evidência nas folhas sadias se move mais que meia faixa entre quase-evento e evento, o
  perfil fica preso à cabeça anterior e **o Rover recusa a nova** até o `fit_edge`
  recalibrar (seção 23) — limiar medido com uma cabeça não vale para outra;
* **o que alimenta**: os quadros que a percepção ativa marca `for_validation` (seção 23) são
  os que o agrônomo deve ver primeiro — é dúvida do modelo, o exemplo que mais ensina;
* **consentimento**: `esquece --fonte <talhão>` tira da memória, dos conjuntos e das rodadas.

**Medido (15/10/2026, embeddings reais, BRACOL como fazenda nova em 3 rodadas, 3
sementes)**: café 0,968 → **0,967 ± 0,003**; BRACOL 0,300 → **0,847 ± 0,015**; alarme falso
em folha sadia do BRACOL 100% → **0%**, do café 2,6% → **1,2%**; acervo externo (só mede)
0,165 → **0,841**. Sem o portão, o ajuste ingênuo leva o café a **0,813**; o portão recusou
9/9 dessas rodadas e 3/3 com rótulos ruins.

## 25. Nível 4: as prioridades da missão na percepção ativa

A missão assinada pode trazer `prioridades` (ver `docs/MISSIONS.md` e `docs/ENXAME.md`): os
trechos de rua da missão com a prioridade relativa que o gêmeo deu pelo valor da informação.
`rover/valtexa_rover/runtime/prioridades.py` lê o bloco (vencido ou malformado: sem mapa, nada
cai) e o Core cruza, a cada quadro, o lado da câmera, a rua daquele lado e o metro do eixo
(navegação) com o mapa.

Na percepção ativa (`percepcao_ativa` no `rover.json`):

* trecho abaixo de `prioridade_minima` (0,15): dúvida comum não ganha vista extra
  (`skipped: prioridade_baixa`); anomalia ganha sempre;
* trecho abaixo de `prioridade_alta` (0,5) não gasta as últimas `reserva_prioridade_alta` (2)
  vistas por minuto da câmera — vale a cada vista, não só na primeira
  (`skipped: orcamento_reservado`);
* a prioridade usada vai no bloco `percepcao` do resultado.

Só o **quanto** insistir muda. A decisão por quadro e os recortes que sobem para a fusão por
planta são os mesmos com ou sem prioridade: a prioridade é o prior fundido do enxame, e
devolvê-lo como evidência contaria a mesma observação duas vezes. Testes:
`rover/tests/test_prioridades.py`.

## 26. Nível 2, o que faltava: dúvida conformal, reobservar, luz, lesão em 3D e procedência

O §23 tinha a percepção ativa disparada pela evidência e pela anomalia, e o §24 o ciclo
contínuo na nuvem. Faltavam a dúvida estatística e a régua da captura no próprio Rover, a
decisão de reobservar com o robô andando, a luz, a posição da lesão na planta e a garantia de
que a cabeça adaptada que chega ao Rover é a que o agrônomo liberou.

**No motor (`edge_ai/engine.py`).** O pacote pode trazer `<sha>.quality.json` (a régua da
captura do `fit_quality`) e `<sha>.conformal.json` (os limiares do `fit_conformal`), com o
código vendorizado byte a byte do `ia/` (`vendor/quality.py`, `vendor/conformal.py`). Régua de
outra resolução ou conformal de outras classes é recusado e anotado em `engine.avisos` — daria
dúvida falsa. Cada recorte sai com:

* `conformal_set`: o conjunto conformal (modo `plain`, limiares base — os mesmos com que a
  fusão da nuvem divide a evidência);
* `quality`: os problemas de captura (`blurry`, `dark`, `overexposed`…), medidos só nos
  recortes que decidem o quadro e nos mais fortes (`RECORTES_MEDIDOS = 8`): o
  pré-processamento é o gargalo da Jetson;
* `geometria` (com mapa de profundidade): o ponto 3D do centro no referencial da câmera, a
  normal e a **inclinação** da superfície por um plano ajustado aos pixels medidos (só os da
  faixa da planta), o resíduo, `superficie` `plana`/`irregular` (resíduo > 12 mm: várias
  folhas, borda, vão — sem orientação) e `face` **sempre `indeterminada`**: a profundidade dá
  a orientação, não qual face está para a câmera (folha ondulada, vista de lado, a mesma
  normal serve às duas). Decidir adaxial/abaxial pede outro modelo.

O corpo dos recortes (`tiles`) leva `quality`, `conformal_sets` e `geometria` quando medidos.

**Na percepção ativa (`runtime/percepcao.py`).** Dois motivos novos de dúvida:
`conformal_ambiguo` (pelo menos `min_recortes_ambiguos` recortes cujo conjunto tem sadia **e**
doença — dúvida da decisão, vai ao agrônomo) e `qualidade_baixa` (pelo menos
`fracao_qualidade_baixa` dos recortes medidos fora da régua — problema do **sensor**: pede
vista nova, não manda o original ao agrônomo sozinho). E o roteiro da vista nova:

1. **reobservável?** Pela velocidade da navegação, quanto o quadro anda na latência da vista
   (`latencia_vista_s`, 0,4 s): `v·t / (2·d·tan(hfov/2))` da largura. Se a região some da
   foto antes da vista, não adianta insistir (`skipped: fora_do_campo`);
2. **desacelerar** só se `reduzir_velocidade` (desligado por padrão, como `pedir_pausa_s`) e
   a navegação publicar `modo: autonomo`, `reducao_permitida: true` e nada de teleoperação,
   obstáculo ou travessia: pede `fator_velocidade` (0,3) por `duracao_reducao_s` (**1,5 s,
   hipótese** — a medir no campo). No Rover V3 vai em `velocidade.json` ao lado do estado da
   navegação; o protocolo com a navegação ainda precisa ser fechado com quem a escreve;
3. **luz auxiliar** (`ajustar_luz`): escuro sobe a luz `passo_luz`, estourado desce
   (borrão e contraste baixo não se resolvem com luz); volta ao nível anterior no fim, aconteça o que acontecer. `LuzPorArquivo`
   escreve `luz.json` — **não testado em hardware** (o driver do LED está por definir);
4. a vista nova passa pela **mesma régua**: vista fora dela não entra na fusão
   (`quality: ruim`, conta em `vista_ruim`). Foco só se a câmera tiver foco ajustável.

**A lesão na planta (nuvem, `ia/app/plant_fusion.py`).** A fusão guarda, por planta e
passada, cada recorte a favor de uma doença (LLR > 0) no referencial **da planta**: ao longo
da rua (o mesmo `x` da associação), altura (câmera menos o `y` 3D), quanto dentro da copa
(profundidade menos a distância da planta), inclinação e face. `GET
/fields/{talhão}/plants/{planta}/lesoes` agrupa as vistas da mesma passada e reencontra a
lesão entre passadas pela doença e pela distância (10 cm ao longo e na altura, 15 cm na
copa): `nova`, `persistente` ou `nao_revista` (caiu, foi tratada, ou a câmera não pegou). O
SaaS repassa a `geometria` de cada recorte (`services/fusao_borda.py`); o conjunto conformal
da borda **não** vai para a fusão — ela foi calibrada sem ele.

**Procedência da cabeça adaptada.** `adapt_campo promove` exige `--aprovador` e grava
`<pesos>.head.provenance.json`: tipo da atualização, cabeça de partida e nova (SHA-256),
rodada, escopo, hash do laudo do portão e do dado de campo, quem liberou, se foi forçada (com
motivo). O hash dela vai carimbado no perfil (`measured.continuo.procedencia_sha256`). O
Rover (`_confere_procedencia`, com `vendor/atualizacao.py`) recusa carregar cabeça que o
perfil diz ter vindo do contínuo sem a procedência, com outra procedência, de outra cabeça ou
torre, sem aprovador ou com portão reprovado sem liberação forçada. A procedência viaja como
extra do OTA (`EXTRAS`), entra na marca que força recarregar e no evento `MODEL_UPDATED`
(`head_provenance`). Os tipos de atualização (`cabeca`, `estrutura`, `embedding`, `prior`,
`regra_de_decisao`) e o que cada um invalida estão no `ia/README.md`.

Limite conhecido: o registry confere o hash do `.onnx`; os extras (perfil, cabeça,
procedência) são conferidos entre si, não contra o registry. Um extra trocado por inteiro de
forma coerente passaria — fechar isso pede o hash dos extras na designação. Testes:
`rover/tests/test_percepcao_n2.py`, `ia/tests/test_lesoes.py`, `ia/tests/test_atualizacao.py`.

## 27. Orçamento de dados e o caminho sem base (rodada 4)

Desde 18/10/2026 **não existe base de comunicação na fazenda**: cada Rover V3 tem a sua
Starlink e fala HTTPS direto com a plataforma (`hal/local.py` `HttpNetwork`, o padrão do
`hardware.json`). A rede "via base" (§19–20) é legado opcional. Tudo o que o Rover manda
sobe pela Starlink **dele** e é medido: o volume passou a ser requisito
(`docs/RODADA4.md`). Código: `runtime/orcamento.py`; configuração: `orcamento_dados` do
`rover.json`; plataforma: `saas/valtexa/services/orcamento_dados.py`.

**O caminho sem base.** Missão, comando, telemetria, sincronização, OTA e vídeo: Rover →
Starlink → plataforma, com a assinatura de sempre. Sem Starlink, o offline-first (§4)
vale igual: o Rover termina a missão que tem, guarda tudo e sobe quando a conexão voltar.
Nada é realocado sem a plataforma (missão só ela assina). A **parada de segurança** nunca
dependeu de rede (rádio próprio ligado ao MCU) e continua assim. Pacote OTA: cada Rover
baixa o seu (a base não redistribui mais); o download conta no orçamento.

**As quatro peças:**

1. **Contabilidade.** Todo pedido do `DeviceClient` passa pelo `OrcamentoDados.contar`:
   corpo enviado + cabeçalhos, corpo recebido e a sobrecarga de TLS/HTTP estimada
   (`sobrecarga_requisicao_bytes`, 700 B). Tabela `uso_dados` por **dia da fazenda**
   (`fuso_horas`, −3), missão e classe (`registros`, `arquivos`, `telemetria`, `comando`,
   `missao`, `atualizacao`, `video`, `outros`). O teto conta subida **e** descida (o plano
   cobra as duas); só a subida é cortável.
2. **Corte por prioridade.** Nunca cortam: comando, missão, telemetria, OTA/vídeo (que têm
   controle próprio), **todo evento** e tudo de prioridade 1. Os resultados (metadados e
   recortes que a fusão por planta usa — a incidência conta todas as plantas por eles) sobem
   até 100% do teto. Os **arquivos** sobem até a fração da prioridade deles
   (`fracao_por_prioridade`: doença/praga 0,6; anomalia/baixa confiança 0,5; normal 0,4;
   secundário 0,3) — os 40% de cima ficam para os resultados. O que passa **espera** na fila
   (dia novo, teto novo); o cortável (prioridade ≥ 5) que esperou mais que `adiado_max_h`
   (24 h) vira **amostra**: 1 a cada `amostra_adiados_cada` (10) fica, o resto é retido.
   Avisos à plataforma (`DATA_BUDGET`, prioridade 1) ao cruzar 80% e 100% do dia.
3. **Tempestade de eventos.** Janela de `tempestade_janela_s` (120 s) com pelo menos
   `tempestade_min_quadros` (40): fração de quadros em evento ≥ `tempestade_fracao` (0,5)
   liga, < `tempestade_fracao_saida` (0,25) desliga. Ligada: só
   `tempestade_originais_por_janela` (4) originais de evento sobem por janela; os quadros de
   contexto do buffer não sobem; o resto fica **retido no disco**; `EVENT_STORM`
   (prioridade 1, nunca cortado) avisa o começo e o fim. O resultado de cada quadro sobe
   sempre.
4. **Deduplicação por planta.** A planta é (rua do lado da câmera, `x_m` da navegação /
   `espacamento_planta_m`). Só os `k_por_planta` (2) originais de evento de maior confiança
   sobem; o que sai do pódio é retido se ainda não começou a subir (com o contexto dele).
   **Sem a navegação dizer a rua e o metro, não deduplica** (não há como saber a planta).

**Retido não é apagado.** O item fica na fila (`RETIDO`, com o motivo), o arquivo no disco
com a retenção de sempre. A plataforma pede de volta pelo comando `liberar_retidos`
(`{"motivo"?, "record_ids"?, "maximo"?}`) e lê o consumo pelo comando `orcamento`.

**O teto pela plataforma.** `PUT /v1/devices/{id}/orcamento-dados` (papel
`device:manage`, auditado) grava `{bytes_dia, bytes_missao}`; ele desce **dentro da missão
assinada** (`orcamento_dados` do corpo) e vale por cima do `rover.json`, persistido no
Rover. Relatório: `GET /v1/devices/{id}/dados` (o que o Rover contou — vem na saúde da
telemetria, `health.dados`, e no `MISSION_COMPLETED` — e o que a plataforma recebeu inteiro,
por dia; os avisos) e `GET /v1/dados/consumo` (uma linha por robô). No console: cartão
"Dados pela Starlink" na página do robô.

**Compressão dos registros.** `sync.comprimir_registros` manda o lote de `/edge/records` em
gzip (assinado comprimido; a plataforma confere a assinatura nos bytes que trafegaram e
descomprime com teto contra bomba). **Desligado por padrão**: ligar só com a plataforma da
rodada 4 no ar.

### Estimativa do dia (SIMULADA)

`python -m valtexa_rover.sim.orcamento` — 8 câmeras (4 por lado), 1 quadro/s, 10 h, Rover a
0,3 m/s, planta a cada 0,7 m. **Hipóteses, não medidas**: original 12 MP = 2,5 MB, reduzida
12 kB, contexto 180 kB, subida sustentada da Starlink 8 Mbit/s, 2% dos quadros normais em
baixa confiança, 1% para validação com 1,5 vista extra. **Medido no corpo sintético**: um
resultado com 70 recortes e 6 classes tem 12 786 B (`tile_probabilities: full`) ou 6 689 B
(`top`); o gzip de um lote de 200 fica com 0,37 (`full`) e 0,22 (`top`) do tamanho. Cenários:
**normal** (2% das plantas com sinal, detecção 0,8, alarme falso 0,5%), **alta** (10%) e
**tempestade** (o modelo dispara em todo quadro, como no vídeo real de 511/511). "Antes" =
a política anterior (sem teto: a banda da Starlink é o limite e o resto fica na fila);
"depois" = teto padrão de 2 GB/dia. Em GB por dia:

| cenário | gerado (GB) | antes: subiu | antes: na fila | depois: subiu (metadados) | depois: na fila | depois: retido | + gzip: subiu (metadados) | + gzip e recorte só com a líder: subiu (metadados) |
| normal | 53.7 | 36.1 | 17.7 | 2.01 (0.95) | 39.2 | 11.8 | 2.01 (0.85) | 1.62 (0.43) |
| alta | 118.4 | 36.1 | 82.4 | 2.03 (0.92) | 55.6 | 60.8 | 2.01 (0.83) | 1.64 (0.43) |
| tempestade | 1045.7 | 36.1 | 1009.7 | 2.19 (1.14) | 12.4 | 1031.0 | 2.09 (0.91) | 1.69 (0.43) |

Como ler:

* **Antes, a fila nunca esvazia.** Mesmo no cenário normal o Rover gera ~54 GB/dia; a
  Starlink a 8 Mbit/s sobe ~36 GB em 10 h e o resto vira fila para a noite — e na
  tempestade (~1 TB/dia) a fila só cresce. Não havia teto: a fatura seria o que a banda
  deixasse.
* **Depois, sobe ~2 GB/dia** (o teto, mais o que nunca corta), em todos os cenários. Na
  tempestade, ~1 TB fica retido no disco em vez de entrar na fila (226 mil originais pela
  deduplicação, 144 mil pela tempestade); na carga alta, 61 GB retidos pela deduplicação.
* **O que sobra no limite são os metadados.** Os resultados com probabilidade de todas as
  classes em cada recorte somam 3,7 GB/dia — sozinhos passam do teto padrão. Com gzip
  (`comprimir_registros`) caem para 1,4 GB e ainda não cabem nos 40% reservados; com gzip e
  `tile_probabilities: top` no perfil (só a classe líder de cada recorte) ficam em ~0,43 GB
  e cabem inteiros. A troca para `top` mexe no que a fusão por planta recebe: decisão do
  Pedro, medida antes contra a calibração da fusão (o perfil e o `fit_fusion` não foram
  tocados aqui).
* O que espera ("na fila") é sobretudo original de baixa confiança e vista de validação: no
  dia seguinte, a amostra (1 a cada 10) e a retenção seguram o crescimento — a simulação não
  aplica a amostra, então essa coluna é o teto do que esperaria.

**Não testado em hardware**: o consumo real da Starlink (cabeçalho e TLS são estimados);
os tamanhos de JPEG da OAK-D; a taxa de evento real. Testes: `rover/tests/test_orcamento.py`
(18), `saas/tests/test_orcamento_dados.py` (6), `saas/web/src/lib/dados.test.ts` (5).

## 28. Inspeção adaptativa orientada por incerteza (19/10/2026)

O Rover não trata toda planta igual: cada quadro cai num de quatro casos, e só os dois últimos
gastam recurso a mais (`runtime/percepcao.py`, `caso_de_inspecao`; o caso vai no bloco
`percepcao.caso` e as contagens em `caso_*` das estatísticas):

| caso | quando | o que o Rover faz |
| --- | --- | --- |
| `suficiente` | aparentemente sadia, foto boa | registra e segue |
| `anomalia_confiante` | achado firme, dentro do que o modelo conhece | marca a posição e guarda a evidência (original + sequência, pela retenção) |
| `ambiguo` | hipóteses empatadas (evidência na fronteira, conjunto conformal com sadia e doença) | sequência extra: vista da região, outro ângulo, luz, desacelerar quando a navegação deixa, térmica; não resolveu → agrônomo |
| `informacao_insuficiente` | fora do que a cabeça conhece, foto fora da régua, sem escala | o mesmo; a confiança do modelo não vale até alguém olhar |

**O requisito: reconhecer quando não sabe.** A cabeça do café sempre escolhe uma classe, e
confiança alta não prova nada (em vídeo real, 511 de 511 quadros viraram evento). A novidade da
IA media o embedding da CNN, que o café não usa no Rover. Agora:

* `ia/app/inference/novidade_cabeca.py` (copiado byte a byte em `edge_ai/vendor/`): PCA de 64
  componentes do embedding SigLIP normalizado, Mahalanobis por classe com covariância encolhida,
  limiar no percentil 99 de um conjunto de **calibração** separado do ajuste;
  `training/fit_novidade_cabeca.py` ajusta e mede; `<sha>.novidade.npz` viaja como extra do
  OTA (`EXTRAS`) e o motor confere torre e classes antes de usar;
* o motor reaproveita o embedding que já calcula para a cabeça (custo: uma projeção 768→64 por
  recorte) e grava `novidade` em cada recorte (`tiles.novidade` no corpo do resultado);
* motivo novo de dúvida **de decisão** `fora_da_distribuicao`: 30% dos recortes acima de
  `limiar_novidade` (`fracao_recortes_novos`) ou um recorte que decide a doença acima dele —
  vale **mesmo com confiança alta** (`incerto_em_novidade`).

**Medido (19/10/2026, embeddings reais da torre do café; 64 componentes fixados antes de olhar
os vídeos).** Com o limiar da referência (escore 1, percentil 99 da calibração): teste do café
0,7% dos recortes, acervo externo de folhas de café 0%; no vídeo do YouTube (20 vídeos, 7.665
recortes pelo motor do Rover) 66% dos recortes e 41% dos que a cabeça chamava de doença com
≥ 90%. **Mas isso não separa erro de domínio novo**: as folhas de café que a cabeça não viu
(BRACOL separado), passadas pelo caminho do quadro (redimensiona + JPEG), têm escore mediano
0,96 — o mesmo dos recortes de vídeo com doença confiante (0,97) — e 15 de 48 sairiam "fora".
O ensaio ponta a ponta com essas folhas mostrou o custo: originais guardados dobraram (29 → 58).

Por isso o limiar do Rover é **1,2** (`limiar_novidade`), calibrado no café não visto (máximo
1,105), não no vídeo: 0 de 48 folhas legítimas marcadas; no vídeo, 46% dos quadros, quase tudo
cena que não é folha (dossel 46%, planta inteira 32%, solo 27%, inseto 26%, pessoa 15%) e 11%
dos recortes "folha de café". **A novidade é rede de segurança contra cena fora do domínio, não
contra diagnóstico errado em folha** — esse erro (a ferrugem vista em toda folha de vídeo) a
novidade no embedding não pega; pega dado de campo validado entrando no aprendizado contínuo.
Recalibrar o limiar com quadro de Rover em cafezal.

**Medida complementar: câmera térmica (opcional).** `hal.interfaces.ThermalCamera`; com ela
no hardware e `usar_termica`, no caso ambíguo ou insuficiente o Rover mede a região da dúvida
(percentil 90) contra o resto do quadro térmico (mediana) e grava `percepcao.termica`
(`delta_c`, `calibrado: false`). **Não decide**: a relação temperatura x doença no café não foi
medida, e o alinhamento térmico x RGB é aproximado pela fração da foto. Simulador:
`hal.simulated.SimulatedThermal`. Não testado em hardware.

**Sem parar demais nem gerar dado à toa**: as vistas extras seguem com o orçamento por minuto e
por câmera, a reserva para trecho de prioridade alta (§25), o tempo máximo por quadro e o
orçamento de dados da Starlink (§27); a pausa e a redução de velocidade continuam desligadas
por padrão e só saem com a navegação dizendo que pode. Testes:
`rover/tests/test_inspecao_adaptativa.py`.
