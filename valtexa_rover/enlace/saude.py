"""Saúde de cada enlace, medida pelo próprio protocolo, e a escolha do melhor com histerese.

* **RTT**: eco do batimento (descontado o tempo que o eco ficou parado do outro lado);
  média e variação no estilo do TCP (RFC 6298) dão o RTO das retransmissões.
* **Perda**: o batimento diz quantos pacotes o outro lado recebeu por este enlace; a
  perda é ``1 - Δrecebidos/Δenviados`` entre dois batimentos, em média móvel.
* **Vazão**: bytes de pedaços enviados por este enlace que foram confirmados, por segundo.
* **RSSI/SNR**: quando o rádio expõe (adaptador da base/rover); ``None`` sem leitura.

Estados: ``VIVO`` ⇄ ``MORTO`` com histerese — morre sem pacote válido por ``t_morto_s``;
só volta depois de ``n_volta`` pacotes válidos espaçados por ``t_volta_s`` (um pacote
solto numa rajada não religa o enlace). ``DEGRADADO`` = vivo, com perda ou RTT acima do
limite.
"""

from __future__ import annotations

from dataclasses import dataclass, field

VIVO, DEGRADADO, MORTO = "vivo", "degradado", "morto"


@dataclass
class ParametrosDeSaude:
    t_morto_s: float = 3.0
    n_volta: int = 3
    t_volta_s: float = 1.0
    perda_degradado: float = 0.25
    rtt_degradado_s: float = 1.5
    rto_min_s: float = 0.15
    rto_max_s: float = 8.0
    troca_fator: float = 1.5
    """Só troca de "melhor enlace" se o outro for este tanto melhor..."""
    troca_espera_s: float = 2.0
    """...por este tempo seguido."""
    troca_imediata: float = 4.0
    """Muito melhor (ex.: 20 Mbit/s contra 250 kbit/s) troca na hora."""
    t_suspeito_s: float = 1.0
    """Calado há mais que isto (2 batimentos): o lote (P2-P4) já vai por outro enlace vivo,
    sem esperar o enlace ser dado como morto (``t_morto_s``). P0/P1 seguem por todos."""
    estabilidade_s: float = 5.0
    """Enlace que voltou só vira "melhor" depois de vivo este tempo, vezes (1 + quedas no
    último minuto) — o rádio que oscila ao longo da rua não fica puxando o lote de volta."""
    estabilidade_max_s: float = 30.0


@dataclass
class SaudeDoEnlace:
    nome: str
    banda_bps: float
    p: ParametrosDeSaude = field(default_factory=ParametrosDeSaude)
    srtt: float | None = None
    rttvar: float = 0.0
    perda: float = 0.0
    vazao_bps: float = 0.0
    rssi_dbm: float | None = None
    snr_db: float | None = None
    vivo: bool = False
    ultimo_rx: float | None = None
    _voltando: list[float] = field(default_factory=list)
    enviados: int = 0
    recebidos: int = 0
    _ref_perda: tuple[int, int] | None = None  # (enviados, recebidos pelo outro lado)
    _bytes_conf: float = 0.0
    _t_vazao: float | None = None
    mudou_em: float = 0.0
    quedas: int = 0
    _quedas_em: list[float] = field(default_factory=list)
    descartados: int = 0  # CRC, tag, replay

    # -- eventos ----------------------------------------------------------------

    def rx_valido(self, agora: float) -> None:
        self.recebidos += 1
        self.ultimo_rx = agora
        if self.vivo:
            return
        if self.quedas == 0:  # o primeiro contato liga o enlace na hora
            self._liga(agora)
            return
        espaco = self.p.t_volta_s / max(1, self.p.n_volta - 1)
        if not self._voltando or agora - self._voltando[-1] >= espaco:
            self._voltando.append(agora)
        if len(self._voltando) >= self.p.n_volta:
            self._liga(agora)

    def _liga(self, agora: float) -> None:
        self.vivo = True
        self.mudou_em = agora
        self._voltando.clear()

    def amostra_rtt(self, rtt: float) -> None:
        rtt = max(0.0, rtt)
        if self.srtt is None:
            self.srtt, self.rttvar = rtt, rtt / 2
        else:
            self.rttvar = 0.75 * self.rttvar + 0.25 * abs(self.srtt - rtt)
            self.srtt = 0.875 * self.srtt + 0.125 * rtt

    def relatorio_do_par(self, recebidos_pelo_par: int) -> None:
        """O batimento do outro lado diz quantos pacotes ele recebeu por este enlace."""
        if self._ref_perda is not None:
            d_env = self.enviados - self._ref_perda[0]
            d_rec = recebidos_pelo_par - self._ref_perda[1]
            if d_env >= 4:
                amostra = min(1.0, max(0.0, 1 - d_rec / d_env))
                self.perda = 0.7 * self.perda + 0.3 * amostra
                self._ref_perda = (self.enviados, recebidos_pelo_par)
                return
            if d_env >= 0:
                return
        self._ref_perda = (self.enviados, recebidos_pelo_par)

    def confirmados(self, n_bytes: int, agora: float) -> None:
        self._bytes_conf += n_bytes
        if self._t_vazao is None:
            self._t_vazao = agora
        dt = agora - self._t_vazao
        if dt >= 1.0:
            amostra = self._bytes_conf * 8 / dt
            self.vazao_bps = amostra if self.vazao_bps == 0 else (
                0.7 * self.vazao_bps + 0.3 * amostra)
            self._bytes_conf, self._t_vazao = 0.0, agora

    def tick(self, agora: float) -> bool:
        """Aplica o tempo: ``True`` se o enlace acabou de morrer."""
        if self.vivo and (self.ultimo_rx is None or agora - self.ultimo_rx > self.p.t_morto_s):
            self.vivo = False
            self.mudou_em = agora
            self.quedas += 1
            self._quedas_em = [t for t in self._quedas_em if agora - t < 60.0] + [agora]
            self._voltando.clear()
            return True
        if not self.vivo and self._voltando and agora - self._voltando[-1] > self.p.t_morto_s:
            self._voltando.clear()
        return False

    # -- leitura ----------------------------------------------------------------

    def suspeito(self, agora: float) -> bool:
        return self.vivo and self.ultimo_rx is not None and \
            agora - self.ultimo_rx > self.p.t_suspeito_s

    def estavel(self, agora: float) -> bool:
        recentes = sum(1 for t in self._quedas_em if agora - t < 60.0)
        espera = min(self.p.estabilidade_max_s, self.p.estabilidade_s * (1 + recentes))
        return self.vivo and (self.quedas == 0 or agora - self.mudou_em >= espera)

    @property
    def estado(self) -> str:
        if not self.vivo:
            return MORTO
        if self.perda > self.p.perda_degradado or (self.srtt or 0) > self.p.rtt_degradado_s:
            return DEGRADADO
        return VIVO

    def rto(self) -> float:
        if self.srtt is None:
            return 1.0
        return min(self.p.rto_max_s, max(self.p.rto_min_s, self.srtt + 4 * self.rttvar))

    def nota(self) -> float:
        """Quanto este enlace vale para o lote: vazão útil sobre o RTT. Morto = 0."""
        if not self.vivo:
            return 0.0
        return self.banda_bps * (1 - min(0.95, self.perda)) / (1 + (self.srtt or 0.5))

    def resumo(self) -> dict:
        return {"estado": self.estado, "rtt_ms": None if self.srtt is None else round(
            self.srtt * 1000, 1), "perda": round(self.perda, 3),
            "vazao_bps": round(self.vazao_bps), "banda_bps": self.banda_bps,
            "rssi_dbm": self.rssi_dbm, "snr_db": self.snr_db, "quedas": self.quedas,
            "descartados": self.descartados}


class EscolhaDoMelhor:
    """O melhor enlace para o lote (P2-P4), com histerese para não ficar trocando."""

    def __init__(self, p: ParametrosDeSaude | None = None) -> None:
        self.p = p or ParametrosDeSaude()
        self.atual: str | None = None
        self._candidato: tuple[str, float] | None = None
        self.trocas = 0

    def escolhe(self, saudes: dict[str, SaudeDoEnlace], agora: float) -> str | None:
        vivos = {n: s for n, s in saudes.items() if s.vivo}
        if not vivos:
            self.atual, self._candidato = None, None
            return None
        # calado há 1 s: o lote sai já por outro enlace vivo que está falando
        aptos = {n: s for n, s in vivos.items() if not s.suspeito(agora)}
        if aptos:
            vivos = aptos
        # o que acabou de voltar (e cai toda hora) espera ficar estável para puxar o lote
        estaveis = {n: s for n, s in vivos.items() if s.estavel(agora) or n == self.atual}
        if estaveis:
            vivos = estaveis
        melhor = max(vivos, key=lambda n: vivos[n].nota())
        if self.atual not in vivos:
            if self.atual is not None:
                self.trocas += 1
            self.atual, self._candidato = melhor, None
            return melhor
        nota_atual, nota_melhor = vivos[self.atual].nota(), vivos[melhor].nota()
        if melhor == self.atual or nota_melhor < self.p.troca_fator * nota_atual:
            self._candidato = None
            return self.atual
        if nota_melhor >= self.p.troca_imediata * nota_atual:  # diferença grande: já
            self.atual, self._candidato = melhor, None
            self.trocas += 1
            return melhor
        if self._candidato is None or self._candidato[0] != melhor:
            self._candidato = (melhor, agora)
        elif agora - self._candidato[1] >= self.p.troca_espera_s:
            self.atual, self._candidato = melhor, None
            self.trocas += 1
        return self.atual
