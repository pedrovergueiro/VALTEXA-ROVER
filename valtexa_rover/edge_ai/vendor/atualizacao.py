"""O que uma atualização do modelo mexe, o que ela exige para valer e de onde ela veio.

Nível 2 (5.1): o ciclo contínuo não pode tratar toda mudança como "modelo novo". Cada tipo
mexe numa coisa, invalida coisas diferentes e pede uma prova diferente antes de ir para o
Rover:

=====================  ============================================  ===========================
tipo                   o que muda                                    quem produz
=====================  ============================================  ===========================
``cabeca``             os pesos da cabeça, mesma forma               ``adapt_campo`` (contínuo)
``estrutura``          a forma da cabeça (neurônios, camadas,        ``fit_heads``/``dnaty``
                       atalhos, ativações)                           (treino novo, fora do ciclo)
``embedding``          a torre de visão (outro espaço de embedding)  troca de backbone
``prior``              a prevalência esperada do talhão              ``prior_shift``, gêmeo
``regra_de_decisao``   limiares e tabelas de evidência do perfil     ``fit_edge``, ``fit_fusion``
=====================  ============================================  ===========================

:func:`classifica` diz o tipo comparando a candidata com a instalada (pela forma, não pelo
nome do comando que a gerou). A **procedência** (:func:`procedencia`) é o registro que vai
ao lado da cabeça adaptada (``<pesos>.head.provenance.json``): de que cabeça partiu, com que
dado, que portão passou, quem liberou. O Rover confere com :func:`verifica_procedencia`
antes de carregar (cópia byte a byte em ``valtexa_rover/edge_ai/vendor/``): cabeça adaptada
sem procedência válida não roda.

Só biblioteca padrão: o Rover usa este arquivo sem o resto do ``ia/``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

FORMATO_PROCEDENCIA = 1


@dataclass(frozen=True)
class TipoDeAtualizacao:
    nome: str
    muda: str
    exige: tuple[str, ...]
    """O que precisa ter acontecido antes de valer em produção."""
    invalida: tuple[str, ...]
    """Artefatos medidos com o estado anterior que deixam de valer."""
    continuo_pode: bool
    """O ciclo contínuo (``adapt_campo``) pode produzir e promover este tipo."""


TIPOS: dict[str, TipoDeAtualizacao] = {t.nome: t for t in (
    TipoDeAtualizacao(
        "cabeca", "pesos da cabeça sobre o embedding, mesma forma",
        exige=("dado validado pelo agrônomo", "portão de regressão APROVADA",
               "aprovação nominal", "procedência ao lado da cabeça"),
        invalida=("conformal da API (.conformal.json)",
                  "perfil da borda, se a deriva passar da folga (.edge.json)"),
        continuo_pode=True),
    TipoDeAtualizacao(
        "estrutura", "forma da cabeça: neurônios, camadas, atalhos ou ativações",
        exige=("treino completo (fit_heads) fora do ciclo contínuo",
               "portão de regressão APROVADA", "perfil da borda medido de novo (sempre)",
               "aprovação nominal", "procedência ao lado da cabeça"),
        invalida=("conformal da API", "perfil da borda (sempre)",
                  "memória de replay: logits da destilação vêm da cabeça antiga"),
        continuo_pode=False),
    TipoDeAtualizacao(
        "embedding", "torre de visão: outro espaço de embedding",
        exige=("embeddings recalculados de todo o acervo", "cabeça treinada de novo",
               "memória e conjuntos de regressão refeitos",
               "régua de qualidade, conformal e perfil da borda medidos de novo",
               "motor da Jetson (TensorRT) reconstruído", "aprovação nominal"),
        invalida=("cabeças", "memória de replay", "conjuntos de regressão", "conformal",
                  "régua de qualidade", "perfil da borda", "motor TensorRT"),
        continuo_pode=False),
    TipoDeAtualizacao(
        "prior", "prevalência esperada no talhão (o prior da decisão)",
        exige=("dado do próprio talhão", "limite de divergência do prior de treino"),
        invalida=(),
        continuo_pode=True),
    TipoDeAtualizacao(
        "regra_de_decisao", "limiares e tabelas de evidência que decidem o quadro",
        exige=("medida em conjunto separado (fit_edge/fit_fusion)",
               "sensibilidade e alarme falso no laudo", "aprovação nominal"),
        invalida=("comparações de alarme feitas com a regra antiga",),
        continuo_pode=False),
)}


def _forma(mlp: Any) -> tuple:
    """O que define a forma de uma cabeça (``EmbeddingMLP``), sem os valores dos pesos."""
    return (tuple(mlp.layer_sizes), tuple(mlp.activations),
            tuple(tuple(s) for s in mlp.skips))


def classifica(atual: Any, candidata: Any) -> str:
    """``embedding``, ``estrutura`` ou ``cabeca`` (duas ``DiagnosisHead``)."""
    if atual.encoder != candidata.encoder:
        return "embedding"
    if tuple(atual.class_names) != tuple(candidata.class_names):
        return "estrutura"
    if _forma(atual.mlp) != _forma(candidata.mlp):
        return "estrutura"
    return "cabeca"


def _sha(valor: Any) -> bool:
    return (isinstance(valor, str) and len(valor) == 64
            and all(c in "0123456789abcdef" for c in valor))


def procedencia(*, tipo: str, modelo: str, encoder: str, cabeca: str, cabeca_base: str,
                rodada: str, escopo: str, quando: str, aprovado_por: str, decisao: str,
                motivos: Sequence[str], laudo_sha256: str, dados: Mapping[str, Any],
                motor: Mapping[str, Any] | None = None, forcada: bool = False,
                motivo: str | None = None) -> dict[str, Any]:
    """O registro de procedência da cabeça adaptada (vai ao lado dela)."""
    registro = {
        "formato": FORMATO_PROCEDENCIA, "tipo": tipo, "modelo": modelo, "encoder": encoder,
        "cabeca": cabeca, "cabeca_base": cabeca_base, "rodada": rodada, "escopo": escopo,
        "quando": quando, "aprovado_por": aprovado_por,
        "portao": {"decisao": decisao, "motivos": list(motivos), "laudo_sha256": laudo_sha256},
        "dados": dict(dados), "motor": dict(motor or {}), "forcada": bool(forcada),
        "motivo": motivo,
    }
    problemas = verifica_procedencia(registro, cabeca)
    if problemas:
        raise ValueError("procedência inválida: " + "; ".join(problemas))
    return registro


def verifica_procedencia(registro: Any, sha_cabeca: str, encoder: str | None = None
                         ) -> list[str]:
    """O que impede a cabeça de valer (lista vazia = vale).

    Confere que o registro é desta cabeça (SHA-256 do arquivo), de um tipo que a borda
    aceita, com portão aprovado (ou forçado com motivo escrito), aprovador nominal e o
    laudo e o dado identificados por hash.
    """
    if not isinstance(registro, Mapping):
        return ["procedência não é um objeto"]
    problemas = []
    if registro.get("formato") != FORMATO_PROCEDENCIA:
        problemas.append(f"formato {registro.get('formato')!r} desconhecido")
    tipo = registro.get("tipo")
    if tipo not in ("cabeca", "estrutura"):
        problemas.append(f"tipo {tipo!r} não é de cabeça")
    if registro.get("cabeca") != sha_cabeca:
        problemas.append("a procedência é de outra cabeça (SHA-256 não confere)")
    if not _sha(registro.get("cabeca_base")):
        problemas.append("sem a cabeça de partida")
    if encoder is not None and registro.get("encoder") != encoder:
        problemas.append("a procedência é de outra torre de visão")
    aprovador = registro.get("aprovado_por")
    if not isinstance(aprovador, str) or not aprovador.strip():
        problemas.append("sem aprovador nominal")
    portao = registro.get("portao")
    if not isinstance(portao, Mapping) or not _sha(portao.get("laudo_sha256")):
        problemas.append("sem o laudo do portão identificado")
    elif portao.get("decisao") != "APROVADA":
        motivo = registro.get("motivo")
        if not (registro.get("forcada") is True and isinstance(motivo, str) and motivo.strip()):
            problemas.append(f"portão {portao.get('decisao')!r} sem liberação forçada com motivo")
    dados = registro.get("dados")
    if not isinstance(dados, Mapping) or not _sha(dados.get("campo_sha256")):
        problemas.append("sem o dado de campo identificado")
    return problemas
