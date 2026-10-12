"""Protocolo de enlace da fazenda (Rover ⇄ base), compartilhado pelo Rover e pela base.

Python puro, sem hardware: cada rádio é uma interface IP e o protocolo fala UDP por todas
ao mesmo tempo. Guia: ``docs/COMUNICACAO_FAZENDA.md``.

| Módulo | O quê |
|---|---|
| :mod:`.quadro` | pacote no fio (CRC32, sessão, contador) e registros FRAG/ACK/BAT |
| :mod:`.cripto` | X25519 + ChaCha20-Poly1305 por salto, aperto de mão, anti-replay |
| :mod:`.fila` | filas persistentes em SQLite WAL, limites, disco cheio |
| :mod:`.saude` | RTT, perda, vazão, RSSI/SNR; vivo/morto com histerese; melhor enlace |
| :mod:`.no` | o nó: ARQ com ACK seletivo, fragmentação, prioridades, redundância, slots |
| :mod:`.rpc` | HTTP sobre o enlace: :class:`~.rpc.RedeViaBase` (HAL de rede do Rover) |
| :mod:`.transferencia` | arquivo grande retomável, em pedaços com hash |
| :mod:`.politica` | perda de enlace no Rover → ``/run/valtexa/enlace.json`` e política |
| :mod:`.emulador` | rádio **emulado** (em processo e por UDP) para os testes de caos |
| :mod:`.transporte` | UDP real e relógios |
"""
