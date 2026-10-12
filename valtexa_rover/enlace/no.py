"""O nó do enlace: várias interfaces de rádio ao mesmo tempo, um ou mais pares.

Um :class:`NoDeEnlace` roda no Rover (iniciador: um par, a base) e outro na base
(respondedor: um par por Rover). O laço é :meth:`NoDeEnlace.passo` — recebe tudo, decide e
envia — e não bloqueia; :mod:`.executor` o roda numa thread em tempo real, e os testes o
chamam com relógio virtual.

**Entrega confiável** (P0-P4, menos os *slots*):

1. :meth:`enviar` grava a mensagem na fila persistente e volta;
2. a mensagem vira pedaços (``FRAG``) do tamanho da menor MTU; cada pedaço tem
   temporizador de retransmissão próprio (RTO do enlace, dobrando a cada tentativa);
3. o receptor confirma com **ACK seletivo** (cumulativo + mapa de 64 bits) e, quando a
   mensagem fica inteira, grava na ``entrada`` + ``vistos`` (mesma transação) e manda
   "entregue inteira"; só aí o emissor a apaga;
4. repetida (retransmissão, duplicação do rádio, cópia redundante por outro enlace, ou
   reenvio depois de um reinício) → ``vistos`` → só confirma de novo;
5. todos os pedaços confirmados e nada de "entregue" (o receptor reiniciou no meio): depois
   de ``prova_s`` o emissor reenvia a mensagem inteira.

**Redundância**: P0 e P1 saem por **todos** os enlaces vivos (temporizador por enlace);
P2-P4 saem pelo **melhor** enlace (:class:`~.saude.EscolhaDoMelhor`, com histerese); sem
enlace vivo, P0/P1 tentam por todos os que têm sessão.

**Agendador** (por enlace, orçamento ``banda_bps`` em balde de fichas): ACK e batimento
primeiro; depois P0, P1 (estritos — nunca passam fome), P2 e P3 intercalados 3:1, e **P4
só com a sobra** (nenhum pedaço de P0-P3 elegível naquele enlace naquele instante).

**Slots** ("só o último vale", teleop): :meth:`enviar_ultimo` substitui o anterior ainda
não entregue, não vai para o disco e morre na validade; o receptor descarta o vencido e o
mais velho que o último entregue (sequência por slot, reservada em blocos no disco para
continuar crescendo depois de um reinício).
"""

from __future__ import annotations

import contextlib
import json
import secrets
import sqlite3
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from valtexa_rover.enlace.cripto import (
    ApertoPendente,
    ChaveDeEnlace,
    FalhaDeCriptoError,
    Sessao,
    conclui_aperto,
    id_do_par,
    inicia_aperto,
    responde_aperto,
)
from valtexa_rover.enlace.fila import Chegada, DiscoCheioError, FilaPersistente
from valtexa_rover.enlace.quadro import (
    ACK_ENTREGUE,
    DADOS,
    HS1,
    HS2,
    R_ACK,
    R_BAT,
    R_FRAG,
    REINICIAR,
    TAM_CABECALHO,
    TAM_CRC,
    TAM_TAG,
    Ack,
    Bat,
    Cabecalho,
    Frag,
    QuadroInvalidoError,
    abre_pacote,
    le_registros,
    monta_pacote,
    registro,
)
from valtexa_rover.enlace.saude import EscolhaDoMelhor, ParametrosDeSaude, SaudeDoEnlace
from valtexa_rover.enlace.transporte import Transporte

P0, P1, P2, P3, P4 = range(5)
NOMES_PRIO = {P0: "P0 segurança/batimento", P1: "P1 comandos/teleop", P2: "P2 telemetria",
              P3: "P3 resultados/eventos", P4: "P4 originais em lote"}
_SOBRA_PACOTE = TAM_CABECALHO + TAM_TAG + TAM_CRC


@dataclass
class ConfigDoEnlace:
    nome: str
    transporte: Transporte
    banda_bps: float
    destino: Any = None
    """Endereço do outro lado (no iniciador). O respondedor aprende pelo pacote autenticado."""
    rajada_bytes: int | None = None


@dataclass
class ParametrosDoNo:
    batimento_s: float = 0.5
    rekey_s: float = 3600.0
    hs_reenvio_s: float = 0.5
    reiniciar_min_s: float = 2.0
    em_voo: dict[int, int] = field(default_factory=lambda: {P0: 64, P1: 64, P2: 32, P3: 32,
                                                           P4: 8})
    peso_p2: int = 3
    peso_p3: int = 1
    rto_max_p01_s: float = 0.5
    remontagem_validade_s: float = 120.0
    lembra_entregues: int = 20000
    saude: ParametrosDeSaude = field(default_factory=ParametrosDeSaude)


@dataclass
class _MsgSaida:
    id: bytes
    prio: int
    canal: int
    corpo: bytes
    expira: float | None
    tam_frag: int
    persistente: bool = True
    slot: str | None = None
    slot_seq: int = 0
    criada: float = 0.0
    total: int = 0
    acked: bytearray = field(default_factory=bytearray)
    n_acked: int = 0
    prox: list[float] = field(default_factory=list)
    prox_link: dict[str, list[float]] = field(default_factory=dict)
    tent: list[int] = field(default_factory=list)
    link_de: list[str | None] = field(default_factory=list)
    todos_em: float | None = None

    def __post_init__(self) -> None:
        self.total = max(1, -(-len(self.corpo) // self.tam_frag))
        self.acked = bytearray(self.total)
        self.prox = [0.0] * self.total
        self.tent = [0] * self.total
        self.link_de = [None] * self.total

    def pedaco(self, i: int) -> bytes:
        return self.corpo[i * self.tam_frag:(i + 1) * self.tam_frag]

    def reinicia(self) -> None:
        self.acked = bytearray(self.total)
        self.n_acked = 0
        self.prox = [0.0] * self.total
        self.prox_link.clear()
        self.todos_em = None


@dataclass
class _Remontagem:
    total: int
    prio: int
    canal: int
    ttl_ms: int | None
    slot: str | None
    slot_seq: int
    partes: dict[int, bytes] = field(default_factory=dict)
    visto: float = 0.0
    expira: float | None = None  # no relógio deste nó, a partir do 1º pedaço que chegou

    def base_e_mapa(self) -> tuple[int, int]:
        base = 0
        while base in self.partes:
            base += 1
        mapa = 0
        for j in range(64):
            if base + j in self.partes:
                mapa |= 1 << j
        return base, mapa


@dataclass
class _Par:
    nome: str
    publica: bytes
    saude: dict[str, SaudeDoEnlace]
    escolha: EscolhaDoMelhor
    destino: dict[str, Any] = field(default_factory=dict)
    sessao: dict[str, Sessao] = field(default_factory=dict)
    aperto: dict[str, ApertoPendente] = field(default_factory=dict)
    em_voo: dict[bytes, _MsgSaida] = field(default_factory=dict)
    recarregar: set[int] = field(default_factory=lambda: set(range(5)))
    remontagem: dict[bytes, _Remontagem] = field(default_factory=dict)
    acks: dict[str, dict[bytes, bool]] = field(default_factory=dict)
    bat_seq: dict[str, int] = field(default_factory=dict)
    bat_prox: dict[str, float] = field(default_factory=dict)
    bat_eco: dict[str, tuple[int, float, float]] = field(default_factory=dict)
    bat_enviados: dict[str, dict[int, float]] = field(default_factory=dict)
    info: dict = field(default_factory=dict)
    slots_tx: dict[str, bytes] = field(default_factory=dict)  # slot -> id em voo
    slots_rx_seq: dict[str, int] = field(default_factory=dict)
    slots_rx: dict[str, Chegada] = field(default_factory=dict)
    reiniciar_em: dict[str, float] = field(default_factory=dict)
    sem_enlace_desde: float | None = None
    inicia: bool = False  # este nó faz o aperto de mão com este par (tem o destino dele)
    info_em: float | None = None  # quando chegou o último batimento com estado


class NoDeEnlace:
    def __init__(self, nome: str, chave: ChaveDeEnlace, fila: FilaPersistente, relogio,
                 enlaces: list[ConfigDoEnlace], pares: dict[str, bytes], *, iniciador: bool,
                 parametros: ParametrosDoNo | None = None,
                 info_batimento: Callable[[], dict] | None = None,
                 destinos: dict[str, dict[str, Any]] | None = None,
                 ao_entregue: Callable[[str, bytes], None] | None = None) -> None:
        """``iniciador``: este nó faz o aperto de mão com os pares (Rover → base). Na malha
        (Parte 13) o papel é **por par**: ``destinos`` = ``{par: {enlace: endereço}}`` dos
        vizinhos com quem este nó inicia (a repetidora inicia com a base e responde ao
        Rover). ``ao_entregue(par, id)``: chamado quando o par confirma "entregue inteira"
        de uma mensagem persistente (a malha usa para saber que o salto guardou)."""
        self.nome = nome
        self.chave = chave
        self.fila = fila
        self.relogio = relogio
        self.iniciador = iniciador
        self.p = parametros or ParametrosDoNo()
        self.enlaces = {e.nome: e for e in enlaces}
        self.info_batimento = info_batimento or (lambda: {})
        self.info_por_par: Callable[[str], dict] | None = None
        """Se definido, o estado do batimento é por vizinho (anúncio de rotas da malha)."""
        self.ao_entregue = ao_entregue
        self._trava = threading.RLock()
        self.novidade = threading.Condition(self._trava)
        """Notificada a cada mensagem que chega inteira (o RPC espera aqui)."""
        mtu = min(e.transporte.mtu for e in enlaces)
        self._mtu_min = mtu
        self._pares: dict[str, _Par] = {}
        self._por_id: dict[bytes, _Par] = {}
        for nome_par, pub in pares.items():
            par = _Par(nome_par, pub,
                       {e.nome: SaudeDoEnlace(e.nome, e.banda_bps, self.p.saude) for e in enlaces},
                       EscolhaDoMelhor(self.p.saude))
            if iniciador:
                for e in enlaces:
                    if e.destino is not None:
                        par.destino[e.nome] = e.destino
            par.destino.update((destinos or {}).get(nome_par, {}))
            par.inicia = bool(par.destino)
            self._pares[nome_par] = par
            self._por_id[id_do_par(pub)] = par
        self._sessoes: dict[bytes, tuple[_Par, str, Sessao]] = {}
        self._pendentes_r: dict[bytes, tuple[_Par, str, Sessao, bytes, Any]] = {}
        self._fichas = {e.nome: 0.0 for e in enlaces}
        self._t_fichas = {e.nome: relogio.agora() for e in enlaces}
        self._entregues: OrderedDict[tuple[str, bytes], None] = OrderedDict()
        # P0/P1 que chegaram com o disco cheio: entregues da memória (o PARAR não pode
        # esperar o disco). seq negativo para não colidir com a tabela ``entrada``.
        self._volateis: list[Chegada] = []
        self._seq_volatil = 0
        # Hora de parede "estável": a do início + o relógio monotônico. Um passo do relógio
        # do sistema (o chrony acertando a hora pelo GNSS, um RTC atrasado no boot) não
        # encurta nem estica a validade do que já está na fila ou no ar.
        self._parede0, self._mono0 = relogio.parede(), relogio.agora()
        fila.parede = self.parede
        self._slot_seq: dict[str, tuple[int, int]] = {}
        self._ultimo_aviso: dict[Any, float] = {}
        self._limpa_vistos_em = 0.0
        self._silencio: dict[tuple[str, str], float] = {}
        """(par, enlace) -> até quando este nó finge que o rádio caiu (ensaio de troca)."""
        self.estat = {"pacotes_tx": 0, "pacotes_rx": 0, "frags_tx": 0, "retransmissoes": 0,
                      "entregues_rx": 0, "repetidas_rx": 0, "invalidos": 0, "cripto": 0,
                      "vencidas_tx": 0, "vencidas_rx": 0, "slot_velho_rx": 0, "apertos": 0,
                      "reinicios_msg": 0, "disco_cheio_rx": 0, "disco_cheio_tx": 0}

    # ------------------------------------------------------------------ API

    def parede(self) -> float:
        """Hora de parede deste nó para validades (imune a passo do relógio do sistema)."""
        return self._parede0 + (self.relogio.agora() - self._mono0)

    def pares(self) -> list[str]:
        return list(self._pares)

    def tam_frag(self, slot: str | None = None) -> int:
        return self._mtu_min - _SOBRA_PACOTE - Frag.sobrecarga(slot)

    def enviar(self, par: str, corpo: bytes, prio: int, canal: int = 0,
               ttl_s: float | None = None) -> bytes:
        """Entrega confiável e persistente. Levanta :class:`~.fila.FilaCheiaError` sem espaço."""
        if not 0 <= prio <= 4:
            raise ValueError("prioridade de 0 a 4")
        msg_id = secrets.token_bytes(16)
        expira = None if ttl_s is None else self.parede() + ttl_s
        with self._trava:
            p = self._pares[par]
            try:
                self.fila.enfileira(msg_id, par, prio, canal, corpo, expira)
            except DiscoCheioError:
                if prio > P1:
                    raise
                # P0/P1 com o disco cheio: vai assim mesmo, só da memória (não sobrevive a
                # um reinício, mas o PARAR e o comando não esperam o disco)
                self.estat["disco_cheio_tx"] += 1
                p.em_voo[msg_id] = _MsgSaida(msg_id, prio, canal, corpo, expira,
                                             self.tam_frag(), persistente=False,
                                             criada=self.relogio.agora())
                return msg_id
            if sum(1 for m in p.em_voo.values() if m.prio == prio) < self.p.em_voo[prio]:
                p.em_voo[msg_id] = _MsgSaida(msg_id, prio, canal, corpo, expira,
                                             self.tam_frag(), criada=self.relogio.agora())
            else:
                p.recarregar.add(prio)
        return msg_id

    def enviar_ultimo(self, par: str, slot: str, corpo: bytes, ttl_s: float, *,
                      prio: int = P1, canal: int = 0) -> None:
        """Só o último vale: substitui o anterior do mesmo slot; não vai para o disco."""
        with self._trava:
            p = self._pares[par]
            antigo = p.slots_tx.pop(slot, None)
            if antigo is not None:
                p.em_voo.pop(antigo, None)
            msg_id = secrets.token_bytes(16)
            m = _MsgSaida(msg_id, prio, canal, corpo, self.parede() + ttl_s,
                          self.tam_frag(slot), persistente=False, slot=slot,
                          slot_seq=self._proxima_seq_slot(slot), criada=self.relogio.agora())
            p.em_voo[msg_id] = m
            p.slots_tx[slot] = msg_id

    def ultimo(self, par: str, slot: str) -> Chegada | None:
        """O último valor do slot que chegou e ainda vale (e o consome)."""
        with self._trava:
            c = self._pares[par].slots_rx.pop(slot, None)
        if c is None or (c.expira is not None and c.expira <= self.parede()):
            return None
        return c

    def chegadas(self, canal: int | None = None, limite: int = 100,
                 excluir_canal: int | None = None) -> list[Chegada]:
        with self._trava:
            volateis = [c for c in self._volateis if (canal is None or c.canal == canal)
                        and c.canal != excluir_canal]
        return volateis[:limite] + self.fila.chegadas(canal, max(0, limite - len(volateis)),
                                                      excluir_canal)

    def slots_recebidos(self, par: str) -> list[str]:
        """Nomes dos *slots* com valor novo esperando (a malha encaminha cada um)."""
        with self._trava:
            return list(self._pares[par].slots_rx)

    def proxima_seq_slot(self, slot: str) -> int:
        with self._trava:
            return self._proxima_seq_slot(slot)

    def consumida(self, seq: int) -> None:
        if seq < 0:
            with self._trava:
                self._volateis = [c for c in self._volateis if c.seq != seq]
            return
        self.fila.consumida(seq)

    def silenciar(self, par: str, enlace: str, segundos: float) -> None:
        """**Ensaio**: o enlace com ``par`` fica mudo (não manda nem aceita nada) por
        ``segundos`` — o efeito de um rádio que morre, sem mexer no rádio. Só para o ensaio
        de troca de enlace (a base só aceita o pedido com ``ensaios_remotos``)."""
        with self._trava:
            self._silencio[(par, enlace)] = self.relogio.agora() + max(0.0, segundos)

    def _mudo(self, par: str, enlace: str, agora: float) -> bool:
        ate = self._silencio.get((par, enlace))
        if ate is None:
            return False
        if agora >= ate:
            del self._silencio[(par, enlace)]
            return False
        return True

    def cancelar(self, par: str, msg_id: bytes) -> bool:
        """Tira da fila do salto uma mensagem ainda não entregue (a malha trocou de rota)."""
        with self._trava:
            p = self._pares[par]
            p.em_voo.pop(msg_id, None)
            return self.fila.entregue(msg_id)

    def vivo(self, par: str) -> bool:
        with self._trava:
            return any(s.vivo for s in self._pares[par].saude.values())

    def info_do_par(self, par: str) -> dict:
        return dict(self._pares[par].info)

    def saude(self) -> dict:
        with self._trava:
            return {
                "no": self.nome,
                "pares": {
                    n: {"enlaces": {e: s.resumo() for e, s in p.saude.items()},
                        "melhor": p.escolha.atual, "trocas": p.escolha.trocas,
                        "vivo": any(s.vivo for s in p.saude.values()),
                        "sem_enlace_desde": p.sem_enlace_desde,
                        "em_voo": len(p.em_voo), "info": p.info}
                    for n, p in self._pares.items()},
                "fila": {NOMES_PRIO[k]: v for k, v in sorted(self.fila.tamanho_saida().items())},
                "descartes": dict(self.fila.descartes),
                "estat": dict(self.estat),
            }

    def fechar(self) -> None:
        for e in self.enlaces.values():
            e.transporte.fechar()
        self.fila.fechar()

    # ------------------------------------------------------------------ laço

    def passo(self) -> None:
        with self._trava:
            agora = self.relogio.agora()
            if agora >= self._limpa_vistos_em:  # a deduplicação lembra 7 dias, não para sempre
                self._limpa_vistos_em = agora + 600.0
                with contextlib.suppress(sqlite3.Error):
                    self.fila.limpa_vistos()
                    self.fila.limpa_entrada_vencida()
                # avisos de "reiniciar" por origem: quem manda lixo de muitos endereços não
                # enche a memória
                self._ultimo_aviso = {k: t for k, t in self._ultimo_aviso.items()
                                      if agora - t < 10.0}
            for nome, e in self.enlaces.items():
                for dados, origem in e.transporte.receber():
                    self._recebe(nome, dados, origem, agora)
            for p in self._pares.values():
                self._manutencao(p, agora)
            for nome in self.enlaces:
                self._repoe_fichas(nome, agora)
                for p in self._pares.values():
                    self._envia(p, nome, agora)

    # ------------------------------------------------------------------ recepção

    def _recebe(self, enlace: str, pacote: bytes, origem: Any, agora: float) -> None:
        self.estat["pacotes_rx"] += 1
        try:
            cab, ad, corpo = abre_pacote(pacote)
        except QuadroInvalidoError:
            self.estat["invalidos"] += 1
            return
        try:
            if cab.tipo == HS1:
                self._recebe_hs1(enlace, cab, ad, corpo, origem, agora)
            elif cab.tipo == HS2:
                self._recebe_hs2(enlace, cab, ad, corpo, agora)
            elif cab.tipo == REINICIAR:
                self._recebe_reiniciar(enlace, cab, agora)
            elif cab.tipo == DADOS:
                self._recebe_dados(enlace, cab, ad, corpo, origem, agora)
        except FalhaDeCriptoError:
            self.estat["cripto"] += 1

    def _recebe_hs1(self, enlace, cab, ad, corpo, origem, agora) -> None:
        pend = self._pendentes_r.get(cab.sessao)
        if pend is not None:  # HS1 repetido (o HS2 se perdeu): o mesmo HS2
            self.enlaces[enlace].transporte.enviar(pend[3], origem)
            return
        if cab.sessao in self._sessoes:
            return
        respondo = {id_do_par(p.publica): p.publica for p in self._pares.values()
                    if not p.inicia}
        if not respondo:
            return
        pub, sessao, hs2 = responde_aperto(self.chave, respondo, cab, ad, corpo, agora)
        par = next(p for p in self._pares.values() if p.publica == pub)
        # no máximo duas pendentes por par e enlace
        velhas = [sid for sid, v in self._pendentes_r.items() if v[0] is par and v[1] == enlace]
        for sid in velhas[:-1]:
            del self._pendentes_r[sid]
        self._pendentes_r[cab.sessao] = (par, enlace, sessao, hs2, origem)
        self.enlaces[enlace].transporte.enviar(hs2, origem)

    def _recebe_hs2(self, enlace, cab, ad, corpo, agora) -> None:
        for par in self._pares.values():
            pend = par.aperto.get(enlace)
            if pend is not None and pend.sid == cab.sessao:
                sessao = conclui_aperto(self.chave, pend, ad, corpo, agora)
                del par.aperto[enlace]
                self._troca_sessao(par, enlace, sessao)
                par.saude[enlace].rx_valido(agora)
                par.bat_prox[enlace] = 0.0  # batimento já: confirma a sessão ao outro lado
                self.estat["apertos"] += 1
                return

    def _troca_sessao(self, par: _Par, enlace: str, sessao: Sessao) -> None:
        antiga = par.sessao.get(enlace)
        par.sessao[enlace] = sessao
        self._sessoes[sessao.sid] = (par, enlace, sessao)
        # a antiga continua decifrando o que ainda está no ar; a de antes dela sai
        for sid, (p2, e2, s2) in list(self._sessoes.items()):
            if p2 is par and e2 == enlace and s2 is not sessao and s2 is not antiga:
                del self._sessoes[sid]

    def _recebe_reiniciar(self, enlace, cab, agora) -> None:
        for par in self._pares.values():
            s = par.sessao.get(enlace)
            if not par.inicia or s is None or s.sid != cab.sessao:
                continue
            ultimo = par.reiniciar_em.get(enlace, -1e9)
            # aviso não autenticado: só vale com o enlace calado há 2 batimentos (num enlace
            # saudável chega batimento a cada 0,5 s; um aviso forjado não derruba a sessão)
            recente = s.ultimo_rx and agora - s.ultimo_rx < 2 * self.p.batimento_s
            if agora - ultimo >= self.p.reiniciar_min_s and not recente:
                par.reiniciar_em[enlace] = agora
                par.aperto.pop(enlace, None)
                par.sessao.pop(enlace, None)

    def _recebe_dados(self, enlace, cab, ad, corpo, origem, agora) -> None:
        achado = self._sessoes.get(cab.sessao)
        if achado is None:
            pend = self._pendentes_r.get(cab.sessao)
            if pend is None:
                self._avisa_reiniciar(enlace, cab.sessao, origem, agora)
                return
            par, enl, sessao, _, _ = pend
            claro = sessao.abre(cab, ad, corpo, agora)  # FalhaDeCriptoError se não for dele
            del self._pendentes_r[cab.sessao]
            self._troca_sessao(par, enl, sessao)
            self.estat["apertos"] += 1
        else:
            par, enl, sessao = achado
            claro = sessao.abre(cab, ad, corpo, agora)
        if enl != enlace:
            return  # sessão de outro enlace chegando por aqui: não mistura saúdes
        if self._silencio and self._mudo(par.nome, enlace, agora):
            return
        if not par.inicia:
            par.destino[enlace] = origem  # roaming: responde para onde veio
        par.saude[enlace].rx_valido(agora)
        try:
            registros = le_registros(claro)
        except QuadroInvalidoError:
            self.estat["invalidos"] += 1
            return
        for r in registros:
            if isinstance(r, Frag):
                self._recebe_frag(par, enlace, r, agora)
            elif isinstance(r, Ack):
                self._recebe_ack(par, r, agora)
            elif isinstance(r, Bat):
                self._recebe_bat(par, enlace, r, agora)

    def _avisa_reiniciar(self, enlace, sid, origem, agora) -> None:
        if all(p.inicia for p in self._pares.values()):
            return
        if agora - self._ultimo_aviso.get(origem, -1e9) < 1.0:
            return
        if len(self._ultimo_aviso) >= 4096:  # inundação de origens: limpa os velhos já
            self._ultimo_aviso = {k: t for k, t in self._ultimo_aviso.items()
                                  if agora - t < 1.0}
            if len(self._ultimo_aviso) >= 4096:
                return
        self._ultimo_aviso[origem] = agora
        self.enlaces[enlace].transporte.enviar(
            monta_pacote(Cabecalho(REINICIAR, sid, 0), b""), origem)

    def _recebe_frag(self, par: _Par, enlace: str, f: Frag, agora: float) -> None:
        acks = par.acks.setdefault(enlace, {})
        chave = (par.nome, f.msg_id)
        rm = par.remontagem.get(f.msg_id)
        if rm is None:
            if chave in self._entregues or (f.slot is None and self.fila.ja_visto(par.nome,
                                                                                 f.msg_id)):
                self._lembra(chave)
                acks[f.msg_id] = True
                self.estat["repetidas_rx"] += 1
                return
            rm = _Remontagem(f.total, f.prio, f.canal, f.ttl_ms, f.slot, f.slot_seq,
                             expira=None if f.ttl_ms is None else self.parede() + f.ttl_ms / 1000)
            par.remontagem[f.msg_id] = rm
        rm.visto = agora
        rm.partes[f.indice] = f.dados
        if len(rm.partes) < rm.total:
            acks.setdefault(f.msg_id, False)
            return
        del par.remontagem[f.msg_id]
        corpo = b"".join(rm.partes[i] for i in range(rm.total))
        parede = self.parede()
        expira = rm.expira
        if expira is not None and expira <= parede:
            acks[f.msg_id] = True
            self._lembra(chave)
            self.estat["vencidas_rx"] += 1
            return
        if rm.slot is not None:
            if rm.slot_seq <= par.slots_rx_seq.get(rm.slot, 0):
                acks[f.msg_id] = True
                self._lembra(chave)
                self.estat["slot_velho_rx"] += 1
                return
            par.slots_rx_seq[rm.slot] = rm.slot_seq
            par.slots_rx[rm.slot] = Chegada(0, f.msg_id, par.nome, rm.prio, rm.canal, corpo,
                                            parede, expira)
        else:
            try:
                novo = self.fila.recebe(par.nome, f.msg_id, rm.prio, rm.canal, corpo, expira)
            except sqlite3.Error:
                # disco cheio (ou com defeito): não confirma "entregue" — o outro lado
                # continua com a mensagem e reenvia; P0/P1 são entregues da memória
                self.estat["disco_cheio_rx"] += 1
                if rm.prio > P1:
                    acks.pop(f.msg_id, None)
                    return
                self._seq_volatil -= 1
                self._volateis.append(Chegada(self._seq_volatil, f.msg_id, par.nome, rm.prio,
                                              rm.canal, corpo, parede, expira))
                novo = True
            if not novo:
                acks[f.msg_id] = True
                self._lembra(chave)
                self.estat["repetidas_rx"] += 1
                return
        acks[f.msg_id] = True
        self._lembra(chave)
        self.estat["entregues_rx"] += 1
        self.novidade.notify_all()

    def _lembra(self, chave) -> None:
        self._entregues[chave] = None
        self._entregues.move_to_end(chave)
        while len(self._entregues) > self.p.lembra_entregues:
            self._entregues.popitem(last=False)

    def _recebe_ack(self, par: _Par, ack: Ack, agora: float) -> None:
        for msg_id, base, mapa in ack.itens:
            m = par.em_voo.get(msg_id)
            if m is None:
                continue
            if base == ACK_ENTREGUE:
                self._concluida(par, m)
                continue
            for i in range(min(base, m.total)):
                self._marca_ack(par, m, i, agora)
            for j in range(64):
                if mapa >> j & 1 and base + j < m.total:
                    self._marca_ack(par, m, base + j, agora)
            if m.n_acked == m.total and m.todos_em is None:
                m.todos_em = agora

    def _marca_ack(self, par: _Par, m: _MsgSaida, i: int, agora: float) -> None:
        if m.acked[i]:
            return
        m.acked[i] = 1
        m.n_acked += 1
        enl = m.link_de[i]
        if enl is not None:
            par.saude[enl].confirmados(len(m.pedaco(i)), agora)

    def _concluida(self, par: _Par, m: _MsgSaida) -> None:
        par.em_voo.pop(m.id, None)
        if m.slot is not None:
            if par.slots_tx.get(m.slot) == m.id:
                del par.slots_tx[m.slot]
            return
        if m.persistente:
            self.fila.entregue(m.id)
            if self.ao_entregue is not None:
                self.ao_entregue(par.nome, m.id)
        par.recarregar.add(m.prio)

    def _recebe_bat(self, par: _Par, enlace: str, b: Bat, agora: float) -> None:
        par.bat_eco[enlace] = (b.seq, b.t_envio, agora)
        enviados = par.bat_enviados.get(enlace, {})
        t = enviados.pop(b.eco_seq, None)
        if t is not None and abs(t - b.eco_t) < 1e-9:
            par.saude[enlace].amostra_rtt(agora - t - b.retido)
        par.saude[enlace].relatorio_do_par(b.recebidos)
        if b.info:
            with contextlib.suppress(ValueError):
                par.info = json.loads(b.info)
                par.info_em = agora

    # ------------------------------------------------------------------ manutenção

    def _manutencao(self, par: _Par, agora: float) -> None:
        for s in par.saude.values():
            s.tick(agora)
        algum = any(s.vivo for s in par.saude.values())
        if algum:
            par.sem_enlace_desde = None
        elif par.sem_enlace_desde is None:
            par.sem_enlace_desde = self.parede()
        par.escolha.escolhe(par.saude, agora)
        if par.inicia:
            self._apertos(par, agora)
        # remontagens abandonadas
        for mid, rm in list(par.remontagem.items()):
            if agora - rm.visto > self.p.remontagem_validade_s:
                del par.remontagem[mid]
        # recarrega da fila persistente o que coube em voo
        if par.recarregar:
            em_voo_ids = set(par.em_voo)
            for prio in sorted(par.recarregar):
                livres = self.p.em_voo[prio] - sum(1 for m in par.em_voo.values()
                                                   if m.prio == prio)
                if livres <= 0:
                    continue
                for s in self.fila.pendentes(par.nome, prio, livres, em_voo_ids):
                    par.em_voo[s.id] = _MsgSaida(s.id, s.prio, s.canal, s.corpo, s.expira,
                                                 self.tam_frag(), criada=agora)
            par.recarregar.clear()
        # vencidas e "todos confirmados sem entregue" (o receptor reiniciou)
        parede = self.parede()
        for m in list(par.em_voo.values()):
            if m.expira is not None and m.expira <= parede:
                par.em_voo.pop(m.id, None)
                self.estat["vencidas_tx"] += 1
                if m.slot is not None:
                    if par.slots_tx.get(m.slot) == m.id:
                        del par.slots_tx[m.slot]
                elif m.persistente:
                    self.fila.descarta(m.id)
                    par.recarregar.add(m.prio)
            elif m.todos_em is not None and agora - m.todos_em > self._prova(par):
                m.reinicia()
                self.estat["reinicios_msg"] += 1

    def _prova(self, par: _Par) -> float:
        rtos = [s.rto() for s in par.saude.values() if s.vivo] or [1.0]
        return max(2.0, 4 * min(rtos))

    def _apertos(self, par: _Par, agora: float) -> None:
        for nome, e in self.enlaces.items():
            destino = par.destino.get(nome)
            if destino is None:
                continue
            s = par.sessao.get(nome)
            precisa = s is None or agora - s.criada > self.p.rekey_s or (
                not par.saude[nome].vivo and agora - (s.ultimo_rx or s.criada)
                > 2 * self.p.saude.t_morto_s)
            if not precisa:
                par.aperto.pop(nome, None)
                continue
            pend = par.aperto.get(nome)
            if pend is not None and agora - pend.enviado_em < self.p.hs_reenvio_s * max(1, min(
                    4, pend.tentativas)):
                continue
            if pend is None or pend.tentativas >= 3:
                pend = inicia_aperto(self.chave, par.publica, agora)
                par.aperto[nome] = pend
            pend.enviado_em = agora
            pend.tentativas += 1
            e.transporte.enviar(pend.pacote, destino)

    # ------------------------------------------------------------------ envio

    def _repoe_fichas(self, enlace: str, agora: float) -> None:
        e = self.enlaces[enlace]
        rajada = e.rajada_bytes or max(4 * e.transporte.mtu, int(e.banda_bps / 8 * 0.05))
        dt = max(0.0, agora - self._t_fichas[enlace])
        self._t_fichas[enlace] = agora
        # 90 % da banda: folga para o buffer do rádio não encher (P1 não espera atrás de P4)
        self._fichas[enlace] = min(rajada, self._fichas[enlace] + 0.9 * e.banda_bps / 8 * dt)

    def _envia(self, par: _Par, enlace: str, agora: float) -> None:
        sessao = par.sessao.get(enlace)
        destino = par.destino.get(enlace)
        if sessao is None or destino is None:
            return
        if self._silencio and self._mudo(par.nome, enlace, agora):
            return
        e = self.enlaces[enlace]
        cabe = e.transporte.mtu - _SOBRA_PACOTE
        pacote: list[bytes] = []
        usado = 0
        prioritario = False  # pacote com controle/P0 pode estourar o balde

        def fecha() -> None:
            nonlocal pacote, usado, prioritario
            if pacote:
                corpo = b"".join(pacote)
                dados = sessao.sela(corpo)
                e.transporte.enviar(dados, destino)
                self._fichas[enlace] -= len(dados)
                par.saude[enlace].enviados += 1
                self.estat["pacotes_tx"] += 1
            pacote, usado, prioritario = [], 0, False

        def poe(reg: bytes, controle: bool = False) -> None:
            nonlocal usado, prioritario
            if usado + len(reg) > cabe:
                fecha()
            pacote.append(reg)
            usado += len(reg)
            prioritario = prioritario or controle

        # 1. ACKs
        acks = par.acks.pop(enlace, {})
        if acks:
            itens = []
            for mid, entregue in acks.items():
                if entregue:
                    itens.append((mid, ACK_ENTREGUE, 0))
                else:
                    rm = par.remontagem.get(mid)
                    if rm is None:
                        continue
                    base, mapa = rm.base_e_mapa()
                    itens.append((mid, base, mapa))
            por_reg = max(1, (cabe - 5) // Ack.TAM_ITEM)
            for k in range(0, len(itens), por_reg):
                poe(registro(R_ACK, Ack(itens[k:k + por_reg]).codifica()), True)
        # 2. batimento
        if agora >= par.bat_prox.get(enlace, 0.0):
            poe(registro(R_BAT, self._batimento(par, enlace, agora)), True)
        # 3. dados, por prioridade
        vivos = [n for n, s in par.saude.items() if s.vivo]
        melhor = par.escolha.atual
        for prio, frags in self._elegiveis(par, enlace, agora, vivos, melhor):
            for m, i in frags:
                f = Frag(m.id, m.prio, m.canal, i, m.total, m.pedaco(i),
                         None if m.expira is None else int(
                             (m.expira - self.parede()) * 1000),
                         m.slot, m.slot_seq)
                reg = registro(R_FRAG, f.codifica())
                if prio > P0 and self._fichas[enlace] - usado - len(reg) - _SOBRA_PACOTE < 0:
                    fecha()
                    return
                poe(reg, prio == P0)
                self._marca_envio(par, m, i, enlace, agora)
        fecha()

    def _batimento(self, par: _Par, enlace: str, agora: float) -> bytes:
        seq = par.bat_seq.get(enlace, 0) + 1
        par.bat_seq[enlace] = seq
        par.bat_prox[enlace] = agora + self.p.batimento_s
        enviados = par.bat_enviados.setdefault(enlace, {})
        enviados[seq] = agora
        for velho in [k for k in enviados if k < seq - 32]:
            del enviados[velho]
        eco_seq, eco_t, quando = par.bat_eco.get(enlace, (0, 0.0, agora))
        estado = (self.info_por_par(par.nome) if self.info_por_par is not None
                  else self.info_batimento())
        info = json.dumps(estado, separators=(",", ":")).encode()
        if len(info) > 800:  # JSON cortado não decodifica: manda vazio e conta
            self.estat["info_grande"] = self.estat.get("info_grande", 0) + 1
            info = b""
        return Bat(seq, agora, eco_seq, eco_t, par.saude[enlace].recebidos, agora - quando,
                   info).codifica()

    def _elegiveis(self, par: _Par, enlace: str, agora: float, vivos: list[str],
                   melhor: str | None):
        """``(prio, [(msg, índice)])`` na ordem do agendador; P4 só sem P0-P3 elegível."""
        redundante = enlace in vivos or not vivos
        lote = enlace == melhor
        por_prio: dict[int, list[tuple[_MsgSaida, int]]] = {k: [] for k in range(5)}
        for m in par.em_voo.values():
            if m.prio <= P1:
                if not redundante:
                    continue
                prox = m.prox_link.setdefault(enlace, [0.0] * m.total)
                for i in range(m.total):
                    if not m.acked[i] and agora >= prox[i]:
                        por_prio[m.prio].append((m, i))
            elif lote:
                for i in range(m.total):
                    if not m.acked[i] and agora >= m.prox[i]:
                        por_prio[m.prio].append((m, i))
        yield P0, por_prio[P0]
        yield P1, por_prio[P1]
        # P2 e P3 intercalados (3:1)
        p2, p3 = por_prio[P2], por_prio[P3]
        mistura, a, b = [], 0, 0
        while a < len(p2) or b < len(p3):
            for _ in range(self.p.peso_p2):
                if a < len(p2):
                    mistura.append(p2[a])
                    a += 1
            for _ in range(self.p.peso_p3):
                if b < len(p3):
                    mistura.append(p3[b])
                    b += 1
        yield P2, mistura
        if not (por_prio[P0] or por_prio[P1] or mistura):
            yield P4, por_prio[P4]

    def _marca_envio(self, par: _Par, m: _MsgSaida, i: int, enlace: str, agora: float) -> None:
        rto = par.saude[enlace].rto()
        if m.tent[i] > 0:
            self.estat["retransmissoes"] += 1
        self.estat["frags_tx"] += 1
        m.link_de[i] = enlace
        # Recuo suave (1,5x, até 4xRTO): o balde de fichas já cadencia o envio, e no rádio a
        # perda é sobretudo aleatória — dobrar sem fim só atrasa o último pedaço.
        recuo = min(4.0, 1.5 ** m.tent[i])
        if m.prio <= P1:
            prox = m.prox_link[enlace]
            prox[i] = agora + min(self.p.rto_max_p01_s, rto * recuo)
        else:
            m.prox[i] = agora + min(self.p.saude.rto_max_s, rto * recuo)
        m.tent[i] += 1

    def _proxima_seq_slot(self, slot: str) -> int:
        atual, teto = self._slot_seq.get(slot, (0, 0))
        if atual >= teto:
            try:
                teto = self.fila.contador(f"slot:{slot}", 1000)
                atual = teto - 1000
            except sqlite3.Error:  # disco cheio: segue na memória (o teleop não para)
                teto = atual + 1000
        atual += 1
        self._slot_seq[slot] = (atual, teto)
        return atual
