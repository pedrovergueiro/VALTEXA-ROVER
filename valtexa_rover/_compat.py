"""Compatibilidade com o Python 3.10 do JetPack 6 (Ubuntu 22.04) da Jetson Orin.

O ONNX Runtime com CUDA/TensorRT que a NVIDIA publica para o JetPack 6 é para o Python
do sistema, o 3.10. ``datetime.UTC`` e ``enum.StrEnum`` só existem a partir do 3.11;
aqui ficam os equivalentes, para o pacote rodar nos dois.
"""

from __future__ import annotations

import sys
from datetime import timezone
from enum import Enum

UTC = timezone.utc

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:  # pragma: no cover — só no 3.10

    class StrEnum(str, Enum):
        """``str(membro)`` e ``f"{membro}"`` devolvem o valor, como o ``StrEnum`` do 3.11."""

        def __str__(self) -> str:
            return str(self.value)

        def __format__(self, spec: str) -> str:
            return format(str(self.value), spec)


__all__ = ["UTC", "StrEnum"]
