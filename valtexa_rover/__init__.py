"""VALTEXA ROVER RUNTIME — o software que roda no computador embarcado do Rover.

Separado da plataforma (``saas/``) e da IA de nuvem (``ia/``): não importa
nenhum dos dois (brief, seção 4). Do ``ia/`` reaproveita, por cópia conferida,
só o pré-processamento e o runtime ONNX (``edge_ai/vendor``).
"""

RUNTIME_VERSION = "0.1.0"
EDGE_AI_VERSION = "0.1.0"
