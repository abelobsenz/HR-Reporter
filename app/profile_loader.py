from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import logging

import jsonschema

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from app.models import ConsultantProfile
from app.utils import load_json

logger = logging.getLogger(__name__)


def _read_profile_file(profile_path: Path) -> Dict[str, Any]:
    suffix = profile_path.suffix.lower()
    with profile_path.open("r", encoding="utf-8") as f:
        if suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("PyYAML is required to load consultant profile YAML")
            payload = yaml.safe_load(f)
        elif suffix == ".json":
            import json

            payload = json.load(f)
        else:
            raise ValueError(f"Unsupported profile format: {suffix}")

    if not isinstance(payload, dict):
        raise ValueError("Consultant profile must be a mapping object")
    return payload


def load_profile_schema() -> Dict[str, Any]:
    schema_path = Path(__file__).resolve().parent / "schemas" / "profile_schema.json"
    return load_json(schema_path)


def load_consultant_profile(profile_path: str | Path) -> ConsultantProfile:
    profile_path = Path(profile_path)
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_path}")

    payload = _read_profile_file(profile_path)
    schema = load_profile_schema()
    jsonschema.validate(instance=payload, schema=schema)
    profile = ConsultantProfile.model_validate(payload)
    logger.info(
        "profile_loaded path=%s name=%s stages=%s expectations=%s",
        profile_path,
        profile.name,
        len(profile.stages),
        len(profile.expectations),
    )
    return profile
