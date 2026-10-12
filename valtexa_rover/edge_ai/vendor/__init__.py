"""Código copiado do ``ia/`` (VALTEXA-B), **byte a byte**, sem edição.

O pacote do Rover não importa o ``ia/``: ele é instalado sozinho no computador
embarcado, sem FastAPI, Keras nem torch. Mas o pré-processamento **precisa** ser
o mesmo do treino — uma divergência não dá erro, só derruba a acurácia em campo
(ver o cabeçalho de ``preprocess.py``). Por isso a cópia é explícita e conferida:

| Arquivo aqui | Origem |
| --- | --- |
| ``preprocess.py`` | ``ia/app/inference/preprocess.py`` |
| ``onnx_runtime.py`` | ``ia/app/onnx_runtime.py`` |
| ``rover_geometry.py`` | ``ia/app/rover.py`` (recortes de tamanho físico fixo) |
| ``heads.py`` | ``ia/app/inference/heads.py`` (cabeça sobre o embedding da torre de visão) |

``tests/test_vendor.py`` compara o SHA-256 das duas pontas quando o ``ia/`` está
ao lado (``ia/`` no repositório da nuvem; ``../VALTEXA-B`` com os repositórios
separados) e falha se alguém mudar um lado sem o outro. Para
atualizar: copie de novo os dois arquivos e rode os testes. Nunca edite aqui.
"""

ORIGENS = {
    "preprocess.py": "ia/app/inference/preprocess.py",
    "onnx_runtime.py": "ia/app/onnx_runtime.py",
    "rover_geometry.py": "ia/app/rover.py",
    "heads.py": "ia/app/inference/heads.py",
}
