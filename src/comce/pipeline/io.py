import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False, default=str)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.{time.time_ns()}.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        stream.write(canonical_json(value) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def append_jsonl(path: Path, value: Any) -> float:
    started = time.perf_counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(canonical_json(value) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    return time.perf_counter() - started


def read_jsonl_strict(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b'\n'):
        raise RuntimeError(f'torn JSONL preserved for adjudication: {path}')
    return [json.loads(line) for line in data.splitlines() if line]
