"""O que todo ensaio tem: parâmetros validados, progresso, cancelamento e um laudo padrão.

Um ensaio **mede**; ele não move o robô. Os que envolvem parada (``parada``,
``homem_morto``) só pedem parada ou observam os arquivos de ``/run/valtexa`` — quem
anda é a navegação, e só com o operador mandando.

Laudo (o que vai à plataforma no resultado do comando ``ensaio``)::

    {"ensaio": "gnss", "fase": 0, "versao": 1, "params": {...},
     "status": "aprovado" | "reprovado" | "nao_avaliado" | "cancelado" | "erro",
     "aprovado": true | false | null,
     "criterios": [{"nome", "valor", "limite", "op", "unidade", "ok"}],
     "medidas": {...},          # o resumo, para a tela
     "amostras": {...},         # os dados brutos: a plataforma refaz as contas com eles
     "simulado": bool, "hardware": "Rover V3 rev E",
     "inicio", "fim", "duracao_s", "erro", "observacao"}

O ``aprovado`` daqui é a prévia do robô (útil no terminal, sem rede). **O que vale é o da
plataforma**, que recalcula com as amostras e com os critérios dela.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from valtexa_rover.clock import Clock
from valtexa_rover.hal.interfaces import Hardware

VERSAO_LAUDO = 1


class EnsaioCancelado(Exception):  # noqa: N818 — é o nome do evento, não de um erro
    """O operador cancelou: o ensaio para no próximo ponto de checagem."""


class ParametroInvalidoError(ValueError):
    """Parâmetro fora do tipo ou da faixa: o ensaio nem começa."""


class EnsaioIndisponivelError(RuntimeError):
    """Falta a peça ou a fonte de dados que o ensaio mede."""


@dataclass(frozen=True)
class Param:
    nome: str
    tipo: type
    padrao: Any = None
    minimo: float | None = None
    maximo: float | None = None
    descricao: str = ""
    obrigatorio: bool = False

    def converte(self, valor: Any) -> Any:
        if valor is None:
            if self.obrigatorio:
                raise ParametroInvalidoError(f"{self.nome}: obrigatório ({self.descricao})")
            return self.padrao
        try:
            if self.tipo is bool:
                if isinstance(valor, str):
                    valor = valor.strip().lower() in ("1", "true", "sim", "s", "yes")
                saida = bool(valor)
            elif self.tipo is list:
                if not isinstance(valor, list | tuple):
                    raise TypeError("esperava uma lista")
                saida = [float(v) for v in valor]
            elif self.tipo is str:
                saida = str(valor)
            else:
                saida = self.tipo(valor)
        except (TypeError, ValueError) as exc:
            raise ParametroInvalidoError(f"{self.nome}: {exc}") from exc
        if isinstance(saida, int | float) and not isinstance(saida, bool):
            if saida != saida:  # NaN
                raise ParametroInvalidoError(f"{self.nome}: não é número")
            if self.minimo is not None and saida < self.minimo:
                raise ParametroInvalidoError(f"{self.nome}: mínimo {self.minimo}")
            if self.maximo is not None and saida > self.maximo:
                raise ParametroInvalidoError(f"{self.nome}: máximo {self.maximo}")
        return saida

    def descreve(self) -> dict[str, Any]:
        return {"nome": self.nome, "tipo": self.tipo.__name__, "padrao": self.padrao,
                "minimo": self.minimo, "maximo": self.maximo, "descricao": self.descricao,
                "obrigatorio": self.obrigatorio}


def criterio(nome: str, valor: float | bool | None, limite: float | bool | None, op: str,
             unidade: str = "") -> dict[str, Any]:
    """Um critério do laudo. ``op``: ``<=``, ``>=``, ``==``. Sem valor medido, reprova."""
    if valor is None or limite is None:
        ok = False
    elif op == "<=":
        ok = valor <= limite
    elif op == ">=":
        ok = valor >= limite
    elif op == "==":
        ok = valor == limite
    else:
        raise ValueError(f"operador {op!r}")
    if isinstance(valor, float):
        valor = round(valor, 4)
    return {"nome": nome, "valor": valor, "limite": limite, "op": op, "unidade": unidade,
            "ok": bool(ok)}


Progresso = Callable[[float, str, dict | None], None]


@dataclass
class Contexto:
    """O que o ensaio recebe: hardware, relógio, fontes extras, progresso e cancelamento.

    ``fontes``: o que não está na HAL — ``carga`` (uma inferência, para o calor),
    ``termica`` (tegrastats), ``rede`` (ida e volta até a plataforma), ``energia``
    (BMS/MCU). Fonte ausente → o ensaio diz o que falta, não inventa número.
    """

    hardware: Hardware
    clock: Clock
    fontes: dict[str, Any] = field(default_factory=dict)
    progresso: Progresso | None = None
    cancelar: threading.Event = field(default_factory=threading.Event)
    passo_s: float = 0.05
    """Maior espera sem olhar o cancelamento (no relógio do sistema)."""

    def checa(self) -> None:
        if self.cancelar.is_set():
            raise EnsaioCancelado()

    def espera(self, segundos: float) -> None:
        """Dorme ``segundos`` no relógio do robô, olhando o cancelamento no caminho."""
        fim = self.clock.monotonic() + max(0.0, segundos)
        while True:
            self.checa()
            falta = fim - self.clock.monotonic()
            if falta <= 0:
                return
            self.clock.sleep(min(falta, self.passo_s))

    def avisa(self, fracao: float, etapa: str, parcial: dict | None = None) -> None:
        self.checa()
        if self.progresso is not None:
            self.progresso(max(0.0, min(1.0, fracao)), etapa, parcial)


class Ensaio:
    """Base. A subclasse define nome, fase, parâmetros e ``executar``."""

    nome: str = ""
    fase: int = 0
    titulo: str = ""
    parametros: tuple[Param, ...] = ()
    hardware_real: str = ""
    """O que este ensaio ainda não fez contra a peça real (vai no catálogo e no laudo)."""

    @classmethod
    def valida(cls, params: dict | None) -> dict[str, Any]:
        params = dict(params or {})
        conhecidos = {p.nome for p in cls.parametros}
        estranhos = sorted(set(params) - conhecidos)
        if estranhos:
            raise ParametroInvalidoError(f"parâmetros desconhecidos: {estranhos}")
        return {p.nome: p.converte(params.get(p.nome)) for p in cls.parametros}

    @classmethod
    def descreve(cls) -> dict[str, Any]:
        return {"nome": cls.nome, "fase": cls.fase, "titulo": cls.titulo,
                "parametros": [p.descreve() for p in cls.parametros],
                "hardware_real": cls.hardware_real}

    def executar(self, ctx: Contexto, p: dict[str, Any]) -> dict[str, Any]:
        """Devolve ``{"criterios": [...], "medidas": {...}, "amostras": {...}}`` e,
        opcionalmente, ``status``/``observacao`` (ex.: ``nao_avaliado``)."""
        raise NotImplementedError


def rodar(ensaio: Ensaio, params: dict | None, ctx: Contexto) -> dict[str, Any]:
    """Valida, executa e monta o laudo. Nunca levanta: erro e cancelamento viram status."""
    inicio_mono, inicio = ctx.clock.monotonic(), ctx.clock.now()
    t0 = time.perf_counter()
    laudo: dict[str, Any] = {
        "ensaio": ensaio.nome, "fase": ensaio.fase, "versao": VERSAO_LAUDO, "params": {},
        "status": "erro", "aprovado": None, "criterios": [], "medidas": {}, "amostras": {},
        "simulado": bool(ctx.hardware.simulated), "hardware": ctx.hardware.hardware_version,
        "inicio": inicio.isoformat(), "fim": None, "duracao_s": None, "erro": None,
        "observacao": ensaio.hardware_real or None,
    }
    try:
        laudo["params"] = p = ensaio.valida(params)
        ctx.avisa(0.0, "iniciando")
        saida = ensaio.executar(ctx, p)
        laudo["criterios"] = saida.get("criterios", [])
        laudo["medidas"] = saida.get("medidas", {})
        laudo["amostras"] = saida.get("amostras", {})
        if saida.get("observacao"):
            laudo["observacao"] = saida["observacao"]
        if saida.get("status") == "nao_avaliado":
            laudo["status"] = "nao_avaliado"
        else:
            laudo["aprovado"] = bool(laudo["criterios"]) and all(
                c["ok"] for c in laudo["criterios"])
            laudo["status"] = "aprovado" if laudo["aprovado"] else "reprovado"
    except EnsaioCancelado:
        laudo["status"] = "cancelado"
    except (ParametroInvalidoError, EnsaioIndisponivelError) as exc:
        laudo["erro"] = str(exc)
    except Exception as exc:  # o ensaio quebrou: o laudo diz onde, o Runtime segue
        laudo["erro"] = f"{type(exc).__name__}: {exc}"
    laudo["fim"] = ctx.clock.now().isoformat()
    laudo["duracao_s"] = round(ctx.clock.monotonic() - inicio_mono, 3)
    laudo["cpu_s"] = round(time.perf_counter() - t0, 3)
    return laudo
