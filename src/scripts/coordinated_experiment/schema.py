import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

PathLike = str | os.PathLike[str]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: PathLike) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be str")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atomic_write(path: PathLike, write: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: PathLike, value: Any) -> None:
    def write(stream: Any) -> None:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")

    _atomic_write(path, write)


def atomic_write_text(path: PathLike, text: str) -> None:
    if not isinstance(text, str):
        raise TypeError("text must be str")
    _atomic_write(path, lambda stream: stream.write(text))


def load_config(path: PathLike) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, Mapping):
        raise ValueError(f"Config must contain a mapping: {path}")
    return dict(value)


def snapshot_hash(snapshot: Mapping[str, Any]) -> str:
    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping")
    return canonical_sha256({
        key: value for key, value in snapshot.items() if key != "snapshot_sha256"
    })
