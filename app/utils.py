from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_json(path: str | Path) -> Dict[str, Any]:
    import json

    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: Any) -> None:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def get_by_dotted_path(data: Dict[str, Any], path: str) -> Any:
    current: Any = data
    for token in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(token)
        if current is None:
            return None
    return current


def flatten_checks_required_fields(checks: Iterable[Dict[str, Any]]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for check in checks:
        for field in check.get("required_fields", []):
            if field not in seen:
                seen.add(field)
                ordered.append(field)
    return ordered


def resolve_repo_path(path_text: str, repo_root: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return Path(repo_root) / path


def unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out
