from __future__ import annotations

from pathlib import Path

import pytest

from app.profile_loader import load_consultant_profile


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_default_profile() -> None:
    profile = load_consultant_profile(REPO_ROOT / "tuning" / "profile.yaml")
    assert profile.name == "Default Startup HR Scale Profile"
    assert len(profile.stages) >= 1
    assert len(profile.expectations) >= 1


def test_stage_expectation_reference_validation(tmp_path: Path) -> None:
    profile_yaml = tmp_path / "broken.yaml"
    profile_yaml.write_text(
        """
version: "1.0"
name: "Broken"
stages:
  - id: s1
    label: "S1"
    min_headcount: 0
    max_headcount: 20
    expectations: [missing_expectation]
expectations:
  - id: existing_expectation
    area: compliance
    claim: "A"
    severity_if_missing: high
    evidence_queries: ["policy"]
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_consultant_profile(profile_yaml)
