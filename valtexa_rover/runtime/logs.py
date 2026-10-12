"""Logging do Rover: JSON por linha, em arquivo rotativo no disco local.

Uma linha por acontecimento, com horário UTC, nível, módulo e mensagem — o que
o técnico abre no campo e o que um coletor manda para a observabilidade
depois. A chave privada nunca é logada (``DeviceIdentity.__repr__`` a esconde).
"""

from __future__ import annotations

import json
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path

from valtexa_rover._compat import UTC


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        linha = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "data", None)
        if extra:
            linha["data"] = extra
        if record.exc_info:
            linha["exc"] = self.formatException(record.exc_info)
        return json.dumps(linha, ensure_ascii=False, default=str)


def setup_logging(data_dir: Path | str, level: int = logging.INFO) -> logging.Logger:
    pasta = Path(data_dir) / "logs"
    pasta.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("valtexa_rover")
    logger.setLevel(level)
    destino = str(pasta / "rover.log")
    if not any(getattr(h, "baseFilename", None) == destino for h in logger.handlers):
        handler = logging.handlers.RotatingFileHandler(
            destino, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    return logger
