"""Двойное логирование: консоль (stdout) + файл logs/<name>.log.

Онлайн-логи пишутся самим процессом в logs/, поэтому работают независимо от
способа запуска (start_agents.sh, cli.py, напрямую). Бенчмарк-логи при этом
пишутся отдельно в logs_benchmark/ (см. common/benchmark_logger.py).
"""

import logging
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"


def setup_logging(name: str, level: int = logging.INFO) -> logging.FileHandler:
    """Настраивает корневой логгер: консоль + файл logs/{name}.log (truncate).

    Возвращает файловый хендлер, чтобы подключать его к сторонним логгерам
    (например, uvicorn).
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level)
    fmt = logging.Formatter(_FORMAT)

    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in root.handlers
    ):
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        root.addHandler(console)

    path = str((LOGS_DIR / f"{name}.log").resolve())
    for h in root.handlers:
        if isinstance(h, logging.FileHandler) and str(Path(h.baseFilename).resolve()) == path:
            return h
    file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)
    return file_handler
