from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import logging

import jsonschema
try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in minimal environments
    yaml = None

from app.models import ConsultantPack, DISCLAIMER_TEXT
from app.utils import load_json

logger = logging.getLogger(__name__)


def _read_pack_file(pack_path: Path) -> Dict[str, Any]:
    suffix = pack_path.suffix.lower()
    with pack_path.open("r", encoding="utf-8") as f:
        if suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError(
                    "PyYAML is required to load YAML packs. Install dependencies from pyproject.toml."
                )
            payload = yaml.safe_load(f)
        elif suffix == ".json":
            import json

            payload = json.load(f)
        else:
            raise ValueError(f"Unsupported pack format: {suffix}")
    if not isinstance(payload, dict):
        raise ValueError("Consultant pack must be a mapping object")
    return payload


def _inject_default_vendor_dictionaries(payload: Dict[str, Any]) -> None:
    prompt_overrides = payload.setdefault("prompt_overrides", {})
    dictionaries = prompt_overrides.setdefault("vendor_dictionaries", {})
    dictionaries.setdefault(
        "hris_vendors",
        ["Workday", "BambooHR", "Rippling", "ADP", "UKG", "SuccessFactors", "Gusto"],
    )
    dictionaries.setdefault(
        "engagement_vendors",
        ["CultureAmp", "Lattice", "Peakon", "Glint", "Officevibe"],
    )
    dictionaries.setdefault(
        "benefits_vendors",
        ["Sequoia", "Nava", "Mercer", "Aon", "HUB International"],
    )
    dictionaries.setdefault(
        "ats_vendors",
        ["Greenhouse", "Lever", "Ashby", "Workable", "SmartRecruiters"],
    )


def _default_question_groups() -> Dict[str, Dict[str, Any]]:
    return {
        "performance_system": {"fields": ["performance.review_cycle", "performance.goal_framework"]},
        "retention_signals": {"fields": ["er_retention.attrition_rate", "er_retention.engagement_survey"]},
        "leave_overtime": {"fields": ["policies.leave_policy", "policies.overtime_policy"]},
        "harassment_tracking": {
            "fields": ["policies.anti_harassment_policy", "hris_data.mandatory_training_tracking"]
        },
    }


def _normalize_pack(payload: Dict[str, Any]) -> Dict[str, Any]:
    for check in payload.get("checks", []):
        if not isinstance(check, dict):
            continue
        if "followup_questions" not in check and check.get("question_if_unknown"):
            check["followup_questions"] = [check["question_if_unknown"]]
        if "retrieval_queries" not in check:
            check["retrieval_queries"] = []
        check.setdefault("evidence_requirements", {})
        check.setdefault("evidence_scoring", {})
        check.setdefault("coverage_expectations", {})

    field_hints = payload.get("field_query_hints", {})
    if isinstance(field_hints, dict):
        for field, cfg in field_hints.items():
            if not isinstance(cfg, dict):
                continue
            cfg.setdefault("followup_questions", [])
            cfg.setdefault("retrieval_queries", [])
            cfg.setdefault("evidence_requirements", {})
            cfg.setdefault("evidence_scoring", {})
            cfg.setdefault("coverage_expectations", {})
            field_hints[field] = cfg
        payload["field_query_hints"] = field_hints

    payload.setdefault("question_groups", _default_question_groups())
    _inject_default_vendor_dictionaries(payload)
    logger.debug(
        "pack_normalized checks=%s field_hints=%s",
        len(payload.get("checks", [])),
        len(payload.get("field_query_hints", {})),
    )
    return payload


def load_pack_schema() -> Dict[str, Any]:
    schema_path = Path(__file__).resolve().parent / "schemas" / "pack_schema.json"
    return load_json(schema_path)


def load_consultant_pack(pack_path: str | Path) -> ConsultantPack:
    pack_path = Path(pack_path)
    if not pack_path.exists():
        raise FileNotFoundError(f"Pack not found: {pack_path}")

    payload = _read_pack_file(pack_path)
    payload = _normalize_pack(payload)
    schema = load_pack_schema()
    jsonschema.validate(instance=payload, schema=schema)
    pack = ConsultantPack.model_validate(payload)

    if pack.disclaimer != DISCLAIMER_TEXT:
        raise ValueError(
            f"Pack disclaimer must be exactly: {DISCLAIMER_TEXT!r}. "
            "This keeps the legal language consistent."
        )

    logger.info("pack_loaded path=%s name=%s checks=%s", pack_path, pack.name, len(pack.checks))
    return pack
