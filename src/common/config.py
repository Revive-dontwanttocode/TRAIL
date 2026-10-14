"""Repository paths, inherited JSON configuration and atomic JSON writes."""

import copy
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def merge(base, changes):
    result = copy.deepcopy(base)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path, _seen=None):
    path = Path(path).resolve()
    seen = set() if _seen is None else _seen
    if path in seen:
        raise ValueError(f"Cyclic config inheritance: {path}")
    seen.add(path)

    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    parent = config.pop("extends", None)
    if parent:
        config = merge(load_config(path.parent / parent, seen), config)
    return config


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
