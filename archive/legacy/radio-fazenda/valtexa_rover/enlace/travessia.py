"""Missão ciente do enlace (Parte 13, item 4): o Rover sabe quando vai entrar numa zona morta.

Com o mapa do levantamento (:class:`~.levantamento.MapaDeCobertura`, o ``cobertura.json``
que o ``valtexa-base levantamento`` gera) e a rota da missão, :class:`MissaoCienteDoEnlace`
olha à frente a cada volta:

1. **Antecipa o envio**: faltando ``antecedencia_s`` para a entrada, pede ao laço do Runtime
   uma sincronização já (o que está na fila sobe enquanto ainda há rádio) e manda à base um
   **aviso de travessia** (P1: zona, duração prevista, hora prevista de saída) — o console da
   base mostra "em travessia" em vez de "caiu", e a base guarda os comandos para o Rover com
   validade que cubra a travessia (o PARAR do console vale ``duração + 60 s``).
2. **Durante**: a política de perda de enlace sabe que a perda é **esperada** (zona mapeada,
   dentro da duração prevista com folga) e aplica a regra da missão para travessia
   planejada (:class:`~.politica.PoliticaDaMissao`); passou muito do previsto, vira perda
   inesperada (pausa).
3. **Registra** cada travessia em ``travessias.jsonl``: prevista contra real, fila na entrada,
   quanto tempo levou para sincronizar tudo na volta.
4. **Ao voltar**: pede sincronização imediata e acompanha até as filas esvaziarem.

**Não comanda motor** e não muda a rota: só escreve estado (``enlace.json`` pela política) e
pede sincronização. **Parar nunca depende disto.** Rota: os ``segments[].path`` da missão
assinada, pontos GeoJSON ``[lon, lat]``; sem rota, projeta em linha reta pela direção e
velocidade do GNSS.

**Simulado / não testado em hardware**: só com o perfil de cafezal do emulador.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from valtexa_rover._compat import UTC
from valtexa_rover.enlace.levantamento import MapaDeCobertura, para_graus, para_metros

LIVRE, ANTECIPANDO, EM_TRAVESSIA, VOLTANDO = "livre", "antecipando", "em_travessia", "voltando"


def rota_da_missao(payload: dict | None) -> list[tuple[float, float]]:
    """(lat, lon) dos ``segments[].path`` (GeoJSON ``[lon, lat]``) da missão assinada."""
    if not payload:
        return []
    corpo = payload.get("command") if isinstance(payload.get("command"), dict) else payload
    pontos: list[tuple[float, float]] = []
    for seg in corpo.get("segments") or []:
        for p in seg.get("path") or []:
            try:
                pontos.append((float(p[1]), float(p[0])))
            except (TypeError, ValueError, IndexError):
                continue
    return pontos


def adiante(rota: list[tuple[float, float]], pos: dict, limite_m: float, desde: int = 0
            ) -> tuple[list[tuple[float, float]], int]:
    """O pedaço da rota à frente do Rover, até ``limite_m``, e o trecho em que ele está.

    Projeta a posição em cada trecho a partir de ``desde`` (o progresso só anda para a
    frente: numa rua de ida e volta, os dois trechos passam pelo mesmo lugar) e pega, entre
    os trechos a até 3 m do mais perto, o que tem a direção do rumo do GNSS (``rumo_deg``);
    sem rumo, o primeiro — e passa ao seguinte quando chega ao fim dele.
    """
    rumo = pos.get("rumo_deg")
    if len(rota) < 2:
        return [(pos["lat"], pos["lon"])], 0
    origem = (pos["lat"], pos["lon"])
    melhor: list[tuple[float, int, float, float]] = []
    for k in range(max(0, desde), len(rota) - 1):
        ax, ay = para_metros(*rota[k], origem)
        bx, by = para_metros(*rota[k + 1], origem)
        dx, dy = bx - ax, by - ay
        n2 = dx * dx + dy * dy
        f = 0.0 if n2 == 0 else max(0.0, min(1.0, -(ax * dx + ay * dy) / n2))
        melhor.append((math.hypot(ax + dx * f, ay + dy * f), k, f, math.sqrt(n2)))
    if not melhor:
        return [origem], desde
    minimo = min(m[0] for m in melhor)
    perto = [m for m in melhor if m[0] <= minimo + 3.0]
    escolha = perto[0]
    if rumo is not None:
        r = math.radians(float(rumo))
        ux, uy = math.sin(r), math.cos(r)
        for m in perto:
            a, b = rota[m[1]], rota[m[1] + 1]
            bx, by = para_metros(*b, a)
            if m[3] and (bx * ux + by * uy) / m[3] > 0.3 and m[2] < 0.999:
                escolha = m
                break
    elif escolha[2] >= 0.999 and len(perto) > 1:
        escolha = perto[1]
    _, k, f, comp = escolha
    saida = [origem]
    andado = comp * (1 - f)
    if andado >= limite_m:
        a, b = rota[k], rota[k + 1]
        g = f + limite_m / comp if comp else 1.0
        bx, by = para_metros(*b, a)
        saida.append(para_graus(bx * g, by * g, a))
        return saida, k
    saida.append(rota[k + 1])
    anterior = rota[k + 1]
    for p in rota[k + 2:]:
        dx, dy = para_metros(p[0], p[1], anterior)
        seg = math.hypot(dx, dy)
        if andado + seg >= limite_m:  # corta o último trecho no limite
            g = (limite_m - andado) / seg if seg else 0.0
            saida.append(para_graus(dx * g, dy * g, anterior))
            break
        andado += seg
        saida.append(p)
        anterior = p
    return saida, k


def reta_a_frente(pos: dict, rumo_deg: float | None, distancia_m: float
                  ) -> list[tuple[float, float]]:
    if rumo_deg is None:
        return [(pos["lat"], pos["lon"])]
    r = math.radians(rumo_deg)
    lat, lon = para_graus(distancia_m * math.sin(r), distancia_m * math.cos(r),
                          (pos["lat"], pos["lon"]))
    return [(pos["lat"], pos["lon"]), (lat, lon)]


@dataclass
class Travessia:
    zona: int
    prevista_entrada_s: float
    prevista_duracao_s: float
    avisada_em: float
    fila_na_entrada: dict = field(default_factory=dict)
    entrou_em: float | None = None
    saiu_em: float | None = None
    sincronizado_em: float | None = None


class MissaoCienteDoEnlace:
    def __init__(self, mapa: MapaDeCobertura | None, *, registro: Path | str,
                 parede, antecedencia_s: float = 20.0, v_padrao_ms: float = 0.8,
                 folga: float = 1.5, folga_s: float = 15.0, avisar=None) -> None:
        """``parede()``: hora (s). ``avisar(dict)``: manda o aviso de travessia à base."""
        self.mapa = mapa
        self.registro = Path(registro)
        self.parede = parede
        self.antecedencia_s = antecedencia_s
        self.v_padrao_ms = v_padrao_ms
        self.folga = folga
        self.folga_s = folga_s
        self.avisar = avisar
        self.estado = LIVRE
        self.atual: Travessia | None = None
        self._sincronizar = False
        self.feitas = 0
        self._trecho = 0
        self._rota_id: int | None = None

    # -- consulta da política e do laço ---------------------------------------------

    def pede_sincronizar(self) -> bool:
        """O laço do Runtime pergunta a cada volta: ``True`` uma vez por pedido."""
        pedido, self._sincronizar = self._sincronizar, False
        return pedido

    def perda_esperada(self) -> tuple[bool, float | None]:
        """``(esperada, segundos que ainda cabem)`` — a política usa para não pausar uma
        travessia planejada. Fora de travessia, ``(False, None)``."""
        t = self.atual
        if t is None or self.estado not in (ANTECIPANDO, EM_TRAVESSIA):
            return False, None
        inicio = t.entrou_em if t.entrou_em is not None else self.parede()
        limite = inicio + t.prevista_duracao_s * self.folga + self.folga_s
        resta = limite - self.parede()
        return resta > 0, max(0.0, resta)

    # -- laço -----------------------------------------------------------------------

    def atualiza(self, pos: dict | None, *, rota: list[tuple[float, float]] | None = None,
                 v_ms: float | None = None, rumo_deg: float | None = None,
                 vivo: bool = True, filas: dict | None = None,
                 vazio: bool = True) -> str:
        """Uma volta (2 Hz). ``vivo``: enlace com a base; ``filas``: tamanho das filas (para
        o registro); ``vazio``: tudo sincronizado (para fechar o registro da volta)."""
        agora = self.parede()
        v = max(0.2, v_ms if v_ms else self.v_padrao_ms)
        if rota is not None and id(rota) != self._rota_id and rota != getattr(
                self, "_rota", None):
            self._rota, self._rota_id, self._trecho = rota, id(rota), 0  # missão nova
        elif rota is not None:
            self._rota_id = id(rota)
        if rota is not None and self.estado != LIVRE:
            _, self._trecho = adiante(rota, pos, 1.0, self._trecho) if pos else (None,
                                                                               self._trecho)
        if self.estado == VOLTANDO:
            if vazio and self.atual is not None:
                self.atual.sincronizado_em = agora
                self._fecha()
            elif self.atual is not None and agora - (self.atual.saiu_em or agora) > 600:
                self._fecha()  # não fica preso: registra sem a hora de sincronizado
            return self.estado
        if pos is None or self.mapa is None:
            return self.estado
        dentro = self.mapa.zona(pos["lat"], pos["lon"])
        if self.estado == LIVRE:
            olhar = v * self.antecedencia_s
            if rota:
                caminho, self._trecho = adiante(rota, pos, olhar, self._trecho)
            else:
                caminho = reta_a_frente(pos, rumo_deg, olhar)
            achou = self.mapa.proxima_zona(caminho) if len(caminho) > 1 else None
            if achou is not None and rota:
                # a entrada está perto: a duração sai da rota inteira que falta (a travessia
                # pode incluir a manobra na cabeceira e a volta pela outra rua)
                longo, _ = adiante(rota, pos, 20_000.0, self._trecho)
                achou = self.mapa.proxima_zona(longo) or achou
            if dentro is not None:
                achou = (dentro, 0.0, self._estima_travessia(dentro, v))
            if achou is not None:
                zona, ate_m, travessia_m = achou
                self.atual = Travessia(zona.id, agora + ate_m / v, travessia_m / v, agora,
                                       dict(filas or {}))
                self.estado = ANTECIPANDO
                self._sincronizar = True  # sobe o que está na fila enquanto há rádio
                if self.avisar is not None:
                    self.avisar({"t": "travessia", "zona": zona.id,
                                 "entrada_s": round(ate_m / v, 1),
                                 "duracao_prevista_s": round(travessia_m / v, 1),
                                 "ate": agora + (ate_m + travessia_m) / v})
        if self.estado == ANTECIPANDO and self.atual is not None:
            if dentro is not None or not vivo:
                self.atual.entrou_em = agora
                self.atual.fila_na_entrada = dict(filas or self.atual.fila_na_entrada)
                self.estado = EM_TRAVESSIA
            elif agora > self.atual.prevista_entrada_s + self.antecedencia_s * 2:
                self.atual = None  # não entrou (mudou de rota): esquece
                self.estado = LIVRE
        elif self.estado == EM_TRAVESSIA and self.atual is not None and vivo and dentro is None:
            self.atual.saiu_em = agora
            self.estado = VOLTANDO
            self._sincronizar = True  # voltou: sincroniza tudo já
        return self.estado

    def _estima_travessia(self, zona, v: float) -> float:
        lado = math.sqrt(len(zona.celulas)) * self.mapa.celula_m
        return lado  # já dentro e sem rota: a escala da zona

    def _fecha(self) -> None:
        t = self.atual
        if t is None:
            return
        real = None if t.entrou_em is None or t.saiu_em is None else t.saiu_em - t.entrou_em
        linha = {
            "zona": t.zona,
            "avisada_em": _iso(t.avisada_em),
            "prevista": {"entrada": _iso(t.prevista_entrada_s),
                         "duracao_s": round(t.prevista_duracao_s, 1)},
            "real": {"entrou": _iso(t.entrou_em), "saiu": _iso(t.saiu_em),
                     "duracao_s": None if real is None else round(real, 1)},
            "fila_na_entrada": t.fila_na_entrada,
            "sincronizado_s_depois_de_sair": None if t.sincronizado_em is None or
            t.saiu_em is None else round(t.sincronizado_em - t.saiu_em, 1),
        }
        self.registro.parent.mkdir(parents=True, exist_ok=True)
        with self.registro.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(linha, ensure_ascii=False) + "\n")
        self.feitas += 1
        self.atual = None
        self.estado = LIVRE


def _iso(t: float | None) -> str | None:
    return None if t is None else datetime.fromtimestamp(t, UTC).isoformat()
