"""As prioridades que a missão traz: onde vale gastar a percepção ativa (Nível 4).

A plataforma prioriza os trechos de rua pelo valor da informação do gêmeo digital
(``ia/app/gemeo/prioridade.py``) e manda, **dentro do corpo assinado da missão**, os trechos
que a missão cobre, com a prioridade relativa (0 a 1) — ver ``saas/valtexa/services/
compartilhamento.py``. Aqui ela vira uma consulta: a rua de um lado da câmera e o metro do
eixo dão a prioridade daquele quadro.

**Só para decidir a bordo.** A prioridade muda **quanto** o Rover insiste num quadro em dúvida
(as vistas extras), nunca **o que** ele conclui: a evidência que sobe para a fusão é a mesma
com ou sem prioridade. Usar o prior fundido como evidência contaria a mesma observação duas
vezes (``uso`` no bloco diz isso).

Mapa ausente, vencido ou malformado: ``None`` — a percepção ativa age como antes, sem mapa. Um
mapa ruim nunca derruba a missão.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

FOLGA_M = 1.0
"""Folga no eixo em volta de cada trecho (o metro da navegação e o da fusão não batem ao
centímetro)."""
MAX_TRECHOS = 2000


@dataclass(frozen=True)
class Trecho:
    rua: str
    x_inicio_m: float
    x_fim_m: float
    prioridade: float


@dataclass(frozen=True)
class MapaDePrioridade:
    plano_id: str | None
    trechos: tuple[Trecho, ...]

    def prioridade(self, rua: str | None, x_m: float | None) -> float | None:
        """A maior prioridade dos trechos da rua que contêm ``x_m``; fora deles, 0 (a missão
        passa por ali, mas não vale insistir). Sem rua ou sem posição: ``None``."""
        if rua is None or x_m is None:
            return None
        melhor = 0.0
        for t in self.trechos:
            if t.rua == rua and t.x_inicio_m - FOLGA_M <= x_m <= t.x_fim_m + FOLGA_M:
                melhor = max(melhor, t.prioridade)
        return melhor


def de_missao(corpo: dict | None, agora: datetime) -> MapaDePrioridade | None:
    """O mapa do corpo da missão (já conferido pela assinatura), se valer ainda."""
    bloco = (corpo or {}).get("prioridades")
    if not bloco:
        return None
    try:
        valido_ate = datetime.fromisoformat(str(bloco["valido_ate"]).replace("Z", "+00:00"))
        if agora > valido_ate:
            logger.info("Prioridades do plano %s vencidas em %s: sem mapa.",
                        bloco.get("plano_id"), valido_ate)
            return None
        trechos = []
        for t in bloco["trechos"][:MAX_TRECHOS]:
            p = float(t["prioridade"])
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"prioridade fora de [0, 1]: {p}")
            for rua in t["ruas"]:
                trechos.append(Trecho(str(rua), float(t["x_inicio_m"]), float(t["x_fim_m"]), p))
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("Prioridades da missão malformadas (%s): a percepção segue sem mapa.", exc)
        return None
    return MapaDePrioridade(bloco.get("plano_id"), tuple(trechos))
