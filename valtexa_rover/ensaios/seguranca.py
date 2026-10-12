"""Segurança que dá para medir por software: latência do ``parar`` e o homem-morto.

**Isto não substitui a parada de emergência física** (botão no robô, botão sem fio, MCU de
segurança), que não depende de software, rede nem nuvem. Aqui se mede só o caminho do
Runtime: pedido de parada → arquivo em ``/run/valtexa`` → a navegação confirma.

Contrato com a pilha de navegação (``navegacao.json``, escrito por ela):

* ``parado_em`` — ISO-8601 UTC: o instante em que a navegação **confirmou as rodas
  paradas** depois do último pedido de parada;
* ``parada_ref`` — o ``at`` do ``parar.json`` que ela obedeceu (é como o ensaio sabe que
  a confirmação é deste pedido e não de um anterior);
* ``v_ms`` (opcional) — a velocidade medida agora, m/s.

Os ensaios **não movem o robô**: ``parada`` só pede parada (o robô fica em ``parado`` no
fim; voltar é decisão do operador); ``homem_morto`` só observa enquanto o operador usa o
controle remoto e solta.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from valtexa_rover.ensaios.base import (
    Contexto,
    Ensaio,
    EnsaioIndisponivelError,
    Param,
    criterio,
)
from valtexa_rover.ensaios.estatistica import percentil


def _instante(texto: Any) -> datetime | None:
    if not isinstance(texto, str):
        return None
    try:
        return datetime.fromisoformat(texto)
    except ValueError:
        return None


class Parada(Ensaio):
    nome = "parada"
    fase = 1
    titulo = "Latência do PARAR até a navegação confirmar"
    parametros = (
        Param("tentativas", int, 5, 1, 100, "pedidos de parada"),
        Param("intervalo_s", float, 2.0, 0.0, 600.0, "espera entre tentativas (s)"),
        Param("latencia_max_ms", float, 500.0, 1.0, 60000.0,
              "latência máxima (p95) até a confirmação (ms)"),
        Param("timeout_s", float, 3.0, 0.1, 120.0, "sem confirmação até aqui, a tentativa falha"),
        Param("exigir_movimento", bool, False,
              descricao="só pede parada com o robô andando (v_ms no navegacao.json)"),
        Param("v_min_ms", float, 0.1, 0.0, 5.0, "velocidade mínima para contar como andando"),
        Param("espera_movimento_s", float, 60.0, 1.0, 3600.0,
              "quanto esperar o operador pôr o robô em movimento (s)"),
    )
    hardware_real = ("simulado/não testado em hardware: depende da pilha de navegação "
                     "publicar parado_em e parada_ref; não mede o freio nem o MCU.")

    def executar(self, ctx: Contexto, p: dict[str, Any]) -> dict[str, Any]:
        controle = ctx.hardware.controller
        if controle is None or not hasattr(controle, "stop"):
            raise EnsaioIndisponivelError("rover sem navegação (hardware.json: navegacao)")
        latencias: list[float | None] = []
        deteccoes: list[float | None] = []
        sem_movimento = 0
        for i in range(p["tentativas"]):
            if p["exigir_movimento"]:
                t_mov = ctx.clock.monotonic()
                while (float((controle.state() or {}).get("v_ms") or 0.0) < p["v_min_ms"]
                       and ctx.clock.monotonic() - t_mov < p["espera_movimento_s"]):
                    ctx.avisa(i / p["tentativas"], f"tentativa {i + 1}: ponha o robô andando")
                    ctx.espera(0.1)
                if float((controle.state() or {}).get("v_ms") or 0.0) < p["v_min_ms"]:
                    sem_movimento += 1
                    latencias.append(None)
                    deteccoes.append(None)
                    continue
            # O mesmo caminho do botão PARAR do centro de comando.
            if hasattr(controle, "set_mode"):
                controle.set_mode("parado")
            pedido = controle.stop(f"ensaio de parada {i + 1}/{p['tentativas']}")
            t0_mono = ctx.clock.monotonic()
            ref = pedido.get("at") if isinstance(pedido, dict) else None
            t0 = _instante(ref) or ctx.clock.now()
            latencia = deteccao = None
            while ctx.clock.monotonic() - t0_mono <= p["timeout_s"]:
                estado = controle.state() or {}
                parado = _instante(estado.get("parado_em"))
                confirma = parado is not None and (
                    estado.get("parada_ref") == ref if ref is not None
                    and estado.get("parada_ref") is not None else parado >= t0)
                if confirma:
                    deteccao = (ctx.clock.monotonic() - t0_mono) * 1000
                    latencia = (parado - t0).total_seconds() * 1000
                    if latencia < 0:  # relógios diferentes: vale a detecção daqui
                        latencia = deteccao
                    break
                ctx.espera(0.01)
            latencias.append(latencia)
            deteccoes.append(deteccao)
            ctx.avisa((i + 1) / p["tentativas"], f"tentativa {i + 1}/{p['tentativas']}",
                      {"latencia_ms": None if latencia is None else round(latencia, 1)})
            if i + 1 < p["tentativas"]:
                ctx.espera(p["intervalo_s"])
        confirmadas = [x for x in latencias if x is not None]
        p95 = percentil(confirmadas, 95) if confirmadas else None
        criterios = [
            criterio("confirmadas_pct", 100.0 * len(confirmadas) / p["tentativas"], 100.0,
                     ">=", "%"),
            criterio("latencia_p95_ms", p95, p["latencia_max_ms"], "<=", "ms"),
        ]
        medidas = {
            "tentativas": p["tentativas"], "confirmadas": len(confirmadas),
            "sem_movimento": sem_movimento,
            "latencia_p95_ms": None if p95 is None else round(p95, 1),
            "latencia_max_ms": round(max(confirmadas), 1) if confirmadas else None,
            "modo_final": "parado",
        }
        return {"criterios": criterios, "medidas": medidas,
                "amostras": {"tentativas": p["tentativas"],
                             "latencias_ms": [None if x is None else round(x, 3)
                                              for x in latencias],
                             "deteccoes_ms": [None if x is None else round(x, 3)
                                              for x in deteccoes]}}


def _le_teleop(caminho) -> dict | None:
    try:
        dados = json.loads(caminho.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return dados if isinstance(dados, dict) else None


class HomemMorto(Ensaio):
    nome = "homem_morto"
    fase = 1
    titulo = "Homem-morto: o teleop.json zera depois do valid_until"
    parametros = (
        Param("episodios", int, 1, 1, 50, "quantas vezes o operador solta o comando"),
        Param("timeout_s", float, 120.0, 1.0, 3600.0, "espera total pelos episódios (s)"),
        Param("tolerancia_s", float, 1.0, 0.0, 10.0,
              "atraso máximo entre o valid_until e o zero (s)"),
    )
    hardware_real = ("simulado/não testado em hardware: mede o Runtime zerando o "
                     "teleop.json; a navegação parar sozinha no valid_until é dela.")

    def executar(self, ctx: Contexto, p: dict[str, Any]) -> dict[str, Any]:
        controle = ctx.hardware.controller
        caminho = getattr(controle, "teleop_path", None)
        if caminho is None:
            raise EnsaioIndisponivelError("rover sem controle remoto (navegação por arquivo)")
        episodios: list[dict[str, Any]] = []
        atual: dict[str, Any] | None = None  # episódio em curso: o operador está mandando
        inicio = ctx.clock.monotonic()
        while (ctx.clock.monotonic() - inicio < p["timeout_s"]
               and sum(1 for e in episodios if e["fim"]) < p["episodios"]):
            ponto = _le_teleop(caminho)
            agora = ctx.clock.now()
            andando = ponto is not None and (abs(float(ponto.get("v") or 0.0)) > 1e-9
                                             or abs(float(ponto.get("w") or 0.0)) > 1e-9)
            if andando:
                ate = _instante(ponto.get("valid_until"))
                if atual is None:
                    atual = {"valid_until": ate, "fim": False, "atraso_s": None,
                             "nav_atraso_s": None}
                    episodios.append(atual)
                elif ate is not None and (atual["valid_until"] is None
                                          or ate > atual["valid_until"]):
                    atual["valid_until"] = ate
            elif atual is not None and ponto is not None:
                # Zerou: o atraso é do último valid_until visto até aqui.
                ate = atual["valid_until"]
                atual["atraso_s"] = None if ate is None else (agora - ate).total_seconds()
                atual["fim"] = True
                v_nav = (controle.state() or {}).get("v_ms") if hasattr(controle,
                                                                        "state") else None
                atual["nav_v_ms"] = v_nav
                atual = None
            ctx.avisa(len([e for e in episodios if e["fim"]]) / p["episodios"],
                      "use o controle remoto e solte", {"episodios": len(episodios)})
            ctx.espera(0.02)
        fechados = [e for e in episodios if e["fim"]]
        atrasos = [e["atraso_s"] for e in fechados if e["atraso_s"] is not None]
        pior = max(atrasos) if atrasos else None
        criterios = [
            criterio("episodios", len(fechados), p["episodios"], ">="),
            criterio("atraso_max_s", pior, p["tolerancia_s"], "<=", "s"),
        ]
        medidas = {"episodios": len(fechados), "abertos": len(episodios) - len(fechados),
                   "atraso_max_s": None if pior is None else round(pior, 3)}
        return {"criterios": criterios, "medidas": medidas,
                "amostras": {"atrasos_s": [round(a, 4) for a in atrasos],
                             "abertos": len(episodios) - len(fechados)}}
