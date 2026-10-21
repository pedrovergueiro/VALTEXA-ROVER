# Sincronização Rover → plataforma

Brief, seções 12, 18, 19, 38 e 52. Os dois lados: o **Sync Engine** do Rover
(`rover/valtexa_rover/runtime/sync.py`) e as rotas do SaaS
(`saas/valtexa/api/routers/edge_sync.py` + `saas/valtexa/services/edge_sync.py`). Todas as
rotas `/v1/device/**` exigem a assinatura Ed25519 do protocolo
(`saas/docs/DEVICE_PROTOCOL.md`, seção 4) — nada novo de autenticação.

## 1. A fila no Rover

Tabela `sync_queue` do SQLite local. Cada item tem os campos da seção 18: **ID**, **hash**
(SHA-256), **tamanho**, **chunks** (tamanho e total), **progresso** (`chunks_done`,
gravado a cada chunk), **prioridade**, **status** (`PENDENTE`, `ENVIANDO`, `CONCLUIDO`,
`FALHOU`), **tentativas**, próxima tentativa e **horário** de criação.

Dois tipos de item:

| Tipo | O que leva | Rota |
| --- | --- | --- |
| `record` | resultado da Edge AI (`kind=result`) ou evento (`kind=event`) | `POST /v1/device/edge/records`, em lote |
| `file` | imagem original, reduzida ou de contexto | `POST /v1/device/uploads` + chunks + `complete` |

A telemetria fica numa tabela à parte e sobe pelo `POST /v1/device/sync` que já existia,
com o `remaining`.

### Ordem (seção 19)

`ORDER BY priority, seq`: 1 evento crítico → 2 doença/praga → 3 anomalia (e baixa
confiança, e o contexto do buffer) → 4 dados para concluir a missão (início/fim, modelo) →
5 imagens reduzidas NORMAL → 6 secundários (upload iniciado/concluído, enlace). Dentro de um
lote de registros, só itens da mesma prioridade. Teste:
`test_ordem_pela_prioridade_da_secao_19`.

### Erros e *backoff*

| Situação | O que o Rover faz |
| --- | --- |
| sem rede (`NetworkUnavailableError`) | item intacto, sem gastar tentativa; o passo termina; `ROVER_OFFLINE` |
| `5xx`, `401`, `408`, `425`, `429` | tentativa +1, espera `base · 2^(n-1)` (até `backoff_max_s`) com *jitter* de 50–100 % |
| `403` | **para a sincronização inteira** (`sync_blocked`) — o protocolo proíbe insistir |
| outro `4xx` | `FALHOU` + `UPLOAD_FAILED`: não adianta repetir o que a nuvem recusou pelo conteúdo |
| `sync.max_attempts` > 0 | desiste depois de N erros de servidor (padrão 0 = nunca) |

## 2. Registros: `POST /v1/device/edge/records`

```json
{"records": [{"id": "6c16…", "kind": "result", "type": "POSSIVEL_DOENCA",
              "sha256": "<sha256 do JSON canônico de body>", "body": {…}}],
 "remaining": 240}
```

**JSON canônico**: `json.dumps(body, sort_keys=True, separators=(",", ":"),
ensure_ascii=False)` em UTF-8. O servidor recalcula; diferente → `rejected`.

Resposta:

```json
{"accepted": ["…"], "duplicated": ["…"], "conflicts": ["…"],
 "rejected": [{"id": "…", "error": "sha256 não confere com o corpo (JSON canônico)"}]}
```

- **mesmo ID + mesmo hash** → `duplicated`, nada muda: o mesmo evento enviado duas vezes é
  **uma** ocorrência (seção 52). Teste: `test_mesmo_evento_duas_vezes_e_uma_ocorrencia`;
- **mesmo ID + outro hash** → `conflicts`, nada muda (não é reenvio: é bug ou corrupção);
- `body.device_id` diferente do robô autenticado → `rejected`;
- talhão ou missão que não são do robô/cliente → gravados como nulos, nunca vinculados;
- corrida entre duas requisições com o mesmo ID → a restrição única
  `(device_id, record_id)` recusa a segunda no *commit*; a rota refaz o lote e o ID volta
  como `duplicated`;
- `remaining` > 0 deixa o robô **SINCRONIZANDO** na frota; 0 grava `last_sync_at`.

Banco: `edge_record` (migração 0018) — só metadados, com as colunas que se filtram
(tipo, missão, talhão, rótulo, confiança, modelo, posição) e o corpo inteiro em `payload`.

## 3. Arquivos em chunks

```
POST /v1/device/uploads            {upload_id, sha256, size_bytes, content_type, chunk_size,
                                    record_id, role, tier, priority}
  → {upload_id, status: receiving|complete, chunk_size, total_chunks, received_chunks: [...]}
PUT  /v1/device/uploads/{id}/chunks/{i}     corpo bruto (opcional X-Chunk-Sha256)
  → {received, total_chunks}
GET  /v1/device/uploads/{id}                estado
POST /v1/device/uploads/{id}/complete       {sha256}
  → {status: complete, asset_id, deduplicated}
```

- **Retomada (queda em 70 % → retoma de 70 %)**: abrir de novo o mesmo `upload_id` devolve
  `received_chunks`; o Rover manda só os que faltam. O progresso também fica no Rover, chunk
  a chunk, e sobrevive a reinício. Testes: `test_queda_em_70_por_cento_retoma_de_70`
  (rover, com reinício no meio; cada chunk enviado exatamente uma vez) e
  `test_upload_com_queda_retoma_do_que_chegou` (SaaS).
- **Reenvio não duplica**: abrir um upload cujo hash o cliente já tem devolve `complete`
  na hora (`deduplicated: true`), sem pedir um byte. O objeto também é endereçado por hash
  no armazenamento. Teste: `test_reenvio_do_mesmo_arquivo_nao_duplica`.
- **Integridade**: cada chunk tem o tamanho exato (o último, o resto); `X-Chunk-Sha256`
  confere o pedaço; no `complete`, o SHA-256 do arquivo montado precisa bater com o
  declarado — senão os chunks são descartados (`422`) e o Rover reenvia.
- **Limites**: chunk de 1 KB a 8 MB; arquivo até `MAX_ASSET_MB` (64); tipos de imagem,
  JSON e `octet-stream`; **cota de armazenamento do plano** aplicada na abertura (`409`
  com a mensagem do plano). Mesmo `upload_id` com outro hash/tamanho → `409`.
- **Isolamento**: o upload é procurado por `(robô autenticado, upload_id)`; o robô de outro
  cliente recebe `404` em tudo. Coberto também pelo teste gerado do OpenAPI
  (`test_isolamento_openapi.py`), que ataca um upload real da vítima.

Banco: `edge_upload` (metadados, ligação com o registro por `record_id` e com o `asset`).
Bytes: `services/storage.py` (`ObjectStore`: disco local em desenvolvimento, S3 em
produção) — o banco nunca guarda imagem.

### *Staging* dos chunks

`services/edge_sync.ChunkStaging` é a interface. `LocalChunkStaging` grava um arquivo por
chunk em `STORAGE_DIR/_uploads` (escrita atômica; a lista do que chegou sai do disco, então
PUTs simultâneos não disputam linha no banco). Serve para desenvolvimento e servidor
único. **Produção com vários servidores** (Parte 9): com `STORAGE_BACKEND=s3`, o
*staging* é `storage.S3ChunkStaging` — cada chunk é um objeto em
`{prefixo}/_staging/edge/{upload}/{índice}`, `received_chunks` é a listagem do prefixo
(paginada), o `complete` monta lendo do bucket, confere o hash e apaga o prefixo. Qualquer
réplica recebe qualquer chunk. **Por que não o multipart nativo** do S3: ele exige partes
de ≥ 5 MB e o Rover manda chunks de 256 KB (retomada barata em rede ruim); o multipart é
usado no objeto final grande. O `complete` é protegido por trava no Redis (duas réplicas
não montam o mesmo upload: a segunda responde 409 com `Retry-After`). Upload abandonado:
o worker apaga em 72 h e a regra de lifecycle `_staging/` em 3 dias. Testes:
`saas/tests/test_producao_s3.py`, `test_coordenacao.py`.

## 4. Leitura pelo cliente

`GET /v1/edge/records?mission_id=&field_id=&kind=&type=&limit=&offset=` (capacidade
`diagnosis:read`, filtrada pelo cliente) — registros com os arquivos de cada um.

## 5. O teste ponta a ponta desta parte

`saas/tests/test_rover_integracao.py`: o Rover (HAL simulada, Edge AI de verdade sobre o
modelo minúsculo) aceita a missão **assinada** pela plataforma, carrega o modelo designado
pelo registry com o hash conferido, roda 10 quadros online, **perde a rede**, roda mais 12,
**reinicia** sem rede no meio da missão, roda mais 8, conclui; a rede volta e **cai de novo
no meio de um arquivo**; volta; tudo sincroniza. Confere: todos os registros do Rover no
SaaS uma vez só, 3 `IMAGE_EVENT_DETECTED`, os arquivos conferidos por hash no object
storage, telemetria, versões na frota, `last_sync_at`, a leitura pelo cliente — e um
reenvio forçado de tudo que **não cria nenhuma linha nova**.

## 6. Parte 5: o que mudou

- **Ponte Edge AI → IA de nuvem.** Quando o resultado de um evento (POSSÍVEL DOENÇA/PRAGA,
  ANOMALIA) **e** o original dele estão na plataforma (a ordem não importa), o evento vira
  `Observation` e passa pelo pipeline de percepção da nuvem (dupla validação, diagnóstico,
  decisão). `edge_record.observation_id` (migração 0021) liga os dois e impede repetir. Roda
  numa transação depois da confirmação ao robô: falha na IA não faz o robô reenviar; o
  worker varre o que ficou (`missao_resultado.varrer`). A Edge AI só **prioriza**.
- **Fusão operacional por planta e relatório.** `GET /v1/missions/{id}/plants` (planta a
  planta: quadros, câmeras, categorias da Edge AI, diagnósticos da nuvem, status e
  incidência com intervalo de Wilson), `/report` (seção 47) e `/comparison?baseline_id=`.
- **Limite de taxa por robô** nas cinco rotas desta página: `DEVICE_RATE_PER_MINUTE`
  (1.200/min); acima, `429` com `Retry-After` — o Rover já trata 429 com *backoff*.
- **Upload abandonado**: sem chunk novo há mais de `UPLOAD_TTL_HOURS` (72 h), o worker
  apaga o *staging* e a linha; o robô que voltar reabre do zero.
- **Corridas corrigidas** (achadas no estresse com 6 robôs em paralelo): dois envios
  simultâneos do mesmo conteúdo disputavam o mesmo `.part` no `LocalObjectStore` (agora o
  temporário tem nome único) e o mesmo `asset` (agora com *savepoint*: quem perde a corrida
  fica deduplicado).
- **Sequência do comando por robô.** A plataforma numerava a `sequence` do comando assinado
  por missão (recomeçava em 1); o protocolo manda o robô recusar sequência que não seja maior
  que a do último comando aceito — a segunda missão de um robô era recusada para sempre.
  Agora a sequência é a maior já emitida para o robô + 1.
- **Backoff depois de reinício**: o boot do Rover zera a espera dos itens pendentes (o
  relógio monotônico recomeça a cada boot).
- **Pendente**: *staging* compartilhado para várias réplicas da API. O *multipart* do S3
  exige partes de 5 MB (o chunk do robô vai de 1 KB a 8 MB), então o caminho é guardar cada
  chunk como objeto (`_uploads/<id>/<i>`) e montar no `complete` — não feito.
