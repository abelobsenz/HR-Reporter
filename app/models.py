from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


DISCLAIMER_TEXT = "This is not legal advice; consider reviewing with counsel."


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    snippet: str
    source_id: Optional[str] = None
    doc_id: Optional[str] = None
    start_char: Optional[int] = Field(default=None, ge=0)
    end_char: Optional[int] = Field(default=None, ge=0)
    evidence_recovered_by: Optional[
        Literal["llm", "supplement_rule", "targeted_retrieval", "deterministic"]
    ] = None
    retrieval_score: Optional[float] = None

    @model_validator(mode="after")
    def validate_offsets(self) -> "Evidence":
        if self.start_char is not None and self.end_char is not None and self.end_char < self.start_char:
            raise ValueError("end_char cannot be less than start_char")
        return self


class Citation(Evidence):
    model_config = ConfigDict(extra="forbid")


class EvidenceStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal[
        "present",
        "not_provided_in_sources",
        "explicitly_missing",
        "not_assessed",
    ]
    retrieval_status: Optional[
        Literal[
            "MENTIONED_EXPLICIT",
            "MENTIONED_IMPLICIT",
            "MENTIONED_AMBIGUOUS",
            "NOT_FOUND_IN_RETRIEVED",
            "NOT_RETRIEVED",
        ]
    ] = None
    coverage_note: Optional[str] = None
    retrieval_queries: List[str] = Field(default_factory=list)
    match_count: int = Field(default=0, ge=0)
    citations: List[Citation] = Field(default_factory=list)


class Policies(BaseModel):
    model_config = ConfigDict(extra="forbid")

    employee_handbook: Optional[bool] = None
    anti_harassment_policy: Optional[bool] = None
    leave_policy: Optional[bool] = None
    overtime_policy: Optional[bool] = None
    accommodation_policy: Optional[bool] = None
    data_privacy_policy: Optional[bool] = None
    disciplinary_policy: Optional[bool] = None


class Hiring(BaseModel):
    model_config = ConfigDict(extra="forbid")

    structured_interviews: Optional[bool] = None
    job_architecture: Optional[bool] = None
    background_checks: Optional[bool] = None
    offer_approval: Optional[bool] = None


class Onboarding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    onboarding_program: Optional[bool] = None
    role_plan_30_60_90: Optional[bool] = None


class ManagerEnablement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manager_training: Optional[bool] = None
    one_on_one_cadence: Optional[str] = None


class Performance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_cycle: Optional[str] = None
    goal_framework: Optional[str | bool] = None


class CompLeveling(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pay_bands: Optional[bool] = None
    leveling_framework: Optional[bool] = None
    compensation_review_cycle: Optional[str] = None


class HRISData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hris_system: Optional[str] = None
    data_audit_cadence: Optional[str] = None
    mandatory_training_tracking: Optional[bool] = None


class ERRetention(BaseModel):
    model_config = ConfigDict(extra="forbid")

    er_case_process: Optional[bool] = None
    attrition_rate: Optional[str] = None
    engagement_survey: Optional[bool] = None


class Benefits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    benefits_broker: Optional[bool] = None
    benefits_eligibility_policy: Optional[bool] = None


class CompanyPeopleSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company_name: Optional[str] = None
    headcount: Optional[int] = Field(default=None, ge=0)
    headcount_range: Optional[str] = None
    primary_locations: List[str] = Field(default_factory=list)
    current_priorities: List[str] = Field(default_factory=list)

    policies: Policies = Field(default_factory=Policies)
    hiring: Hiring = Field(default_factory=Hiring)
    onboarding: Onboarding = Field(default_factory=Onboarding)
    manager_enablement: ManagerEnablement = Field(default_factory=ManagerEnablement)
    performance: Performance = Field(default_factory=Performance)
    comp_leveling: CompLeveling = Field(default_factory=CompLeveling)
    hris_data: HRISData = Field(default_factory=HRISData)
    er_retention: ERRetention = Field(default_factory=ERRetention)
    benefits: Benefits = Field(default_factory=Benefits)

    key_risks: List[str] = Field(default_factory=list)
    evidence_map: Dict[str, EvidenceStatus] = Field(default_factory=dict)


class MetricItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str
    value: Optional[str] = None
    status: Literal["found", "unknown"]
    evidence: List[Citation] = Field(default_factory=list)


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_id: str
    area: str
    title: str
    severity: Literal["low", "medium", "high", "critical"]
    evidence_status: Literal[
        "present",
        "not_provided_in_sources",
        "explicitly_missing",
        "not_assessed",
    ]
    retrieval_status: Optional[
        Literal[
            "MENTIONED_EXPLICIT",
            "MENTIONED_IMPLICIT",
            "MENTIONED_AMBIGUOUS",
            "NOT_FOUND_IN_RETRIEVED",
            "NOT_RETRIEVED",
        ]
    ] = None
    needs_confirmation: bool = False
    is_threshold_prompt: bool = False
    stage_reason: str
    evidence: List[Citation] = Field(default_factory=list)
    subchecks: List["Subcheck"] = Field(default_factory=list)
    actions: List[str] = Field(default_factory=list)
    owner: str = "TBD / assign (e.g., HR/People, Finance, Ops)"
    metrics: List[MetricItem] = Field(default_factory=list)
    questions: List[str] = Field(default_factory=list)
    score: Optional[float] = None

    @model_validator(mode="after")
    def validate_evidence_status(self) -> "Finding":
        all_citations = list(self.evidence)
        for subcheck in self.subchecks:
            all_citations.extend(subcheck.citations)

        explicit_absence_patterns = [
            r"\bno\b",
            r"\bnot in place\b",
            r"\bdoes not exist\b",
            r"\bmissing\b",
            r"\bnone\b",
            r"\bwithout\b",
        ]

        if self.evidence_status == "present":
            if len(all_citations) < 1:
                raise ValueError("present evidence_status requires at least one citation")

        if self.evidence_status == "not_provided_in_sources":
            # Allow citations for ambiguous/implicit mentions that still require confirmation.
            if len(all_citations) != 0 and self.retrieval_status not in {
                "MENTIONED_IMPLICIT",
                "MENTIONED_AMBIGUOUS",
            }:
                raise ValueError(
                    "not_provided_in_sources citations are only allowed for implicit/ambiguous retrieval matches"
                )

        if self.evidence_status == "explicitly_missing":
            if len(all_citations) < 1:
                raise ValueError("explicitly_missing requires at least one citation")
            text = " ".join(c.snippet.lower() for c in all_citations)
            if not any(re.search(pattern, text) for pattern in explicit_absence_patterns):
                raise ValueError(
                    "explicitly_missing must include citation text explicitly indicating absence"
                )

        return self


class Subcheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_key: str
    evidence_status: Literal[
        "present",
        "not_provided_in_sources",
        "explicitly_missing",
        "not_assessed",
    ]
    retrieval_status: Optional[
        Literal[
            "MENTIONED_EXPLICIT",
            "MENTIONED_IMPLICIT",
            "MENTIONED_AMBIGUOUS",
            "NOT_FOUND_IN_RETRIEVED",
            "NOT_RETRIEVED",
        ]
    ] = None
    citations: List[Citation] = Field(default_factory=list)
    missing_reason: Optional[str] = None

    @model_validator(mode="after")
    def validate_subcheck(self) -> "Subcheck":
        explicit_absence_patterns = [
            r"\bno\b",
            r"\bnot in place\b",
            r"\bdoes not exist\b",
            r"\bmissing\b",
            r"\bnone\b",
            r"\bwithout\b",
        ]

        if self.evidence_status == "present" and len(self.citations) < 1:
            raise ValueError("present subcheck requires citations")

        if self.evidence_status == "not_provided_in_sources" and len(self.citations) != 0:
            if self.retrieval_status not in {"MENTIONED_IMPLICIT", "MENTIONED_AMBIGUOUS"}:
                raise ValueError(
                    "not_provided_in_sources subcheck citations are only allowed for implicit/ambiguous retrieval matches"
                )

        if self.evidence_status == "explicitly_missing":
            if len(self.citations) < 1:
                raise ValueError("explicitly_missing subcheck requires citations")
            text = " ".join(c.snippet.lower() for c in self.citations)
            if not any(re.search(pattern, text) for pattern in explicit_absence_patterns):
                raise ValueError("explicitly_missing subcheck citations must indicate absence")

        return self


class GrowthArea(Finding):
    model_config = ConfigDict(extra="forbid")


class RiskItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    severity: Literal["low", "medium", "high", "critical"]
    statement: str
    evidence: List[Citation] = Field(default_factory=list)
    mitigation: List[str] = Field(default_factory=list)
    legal_note: str = DISCLAIMER_TEXT


class PlanAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str
    rationale: str
    evidence: List[Citation] = Field(default_factory=list)


class Plan3090(BaseModel):
    model_config = ConfigDict(extra="forbid")

    why_now: str
    days_30: List[PlanAction] = Field(default_factory=list)
    days_60: List[PlanAction] = Field(default_factory=list)
    days_90: List[PlanAction] = Field(default_factory=list)


class FinalReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: str
    confidence: float = Field(ge=0, le=1)
    drivers: List[str] = Field(default_factory=list)
    top_growth_areas: List[Finding] = Field(default_factory=list)
    risks: List[RiskItem] = Field(default_factory=list)
    plan_30_60_90: Plan3090
    follow_up_questions: List[str] = Field(default_factory=list)
    unknowns: List[str] = Field(default_factory=list)
    reviewed_sources: List[str] = Field(default_factory=list)
    coverage_summary: Dict[str, Any] = Field(default_factory=dict)
    retrieval_summary: Dict[str, Any] = Field(default_factory=dict)
    stage_note: Optional[str] = None
    coverage_note: Optional[str] = None
    disclaimer: str = DISCLAIMER_TEXT


class RawDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    source_id: Optional[str] = None
    source: str
    source_type: Literal["text", "file", "url"]
    canonical_url: Optional[str] = None
    final_url: Optional[str] = None
    retrieved_at: Optional[str] = None
    title: Optional[str] = None
    content_hash: Optional[str] = None
    http_status: Optional[int] = None
    content_type: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    text: str


class TextChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    doc_id: str
    section: str
    heading_path: List[str] = Field(default_factory=list)
    is_list: bool = False
    is_table: bool = False
    start_char: Optional[int] = Field(default=None, ge=0)
    end_char: Optional[int] = Field(default=None, ge=0)
    nav_score: Optional[float] = None
    text: str


class StageBand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    min_headcount: int = Field(ge=0)
    max_headcount: Optional[int] = Field(default=None, ge=0)
    description: str


class CheckDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    area: str
    title: str
    description: str
    severity: Literal["low", "medium", "high", "critical"]
    required_fields: List[str] = Field(default_factory=list)
    actions: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)
    question_if_unknown: Optional[str] = None
    followup_questions: List[str] = Field(default_factory=list)
    retrieval_queries: List[str] = Field(default_factory=list)
    evidence_requirements: Dict[str, Any] = Field(default_factory=dict)
    evidence_scoring: Dict[str, Any] = Field(default_factory=dict)
    coverage_expectations: Dict[str, Any] = Field(default_factory=dict)
    question_group: Optional[str] = None
    applies_to_stages: List[str] = Field(default_factory=list)
    compliance: bool = False


class WeightsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    area_weights: Dict[str, float]
    severity_weights: Dict[str, float]
    compliance_multiplier: float = Field(default=1.25, ge=1.0)
    unknown_penalty: float = Field(default=0.1, ge=0, le=1)


class TemplatesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    markdown_template: str
    section_titles: Dict[str, str] = Field(default_factory=dict)


class ConsultantPack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    name: str
    disclaimer: str = DISCLAIMER_TEXT
    stages: List[StageBand]
    weights: WeightsConfig
    checks: List[CheckDefinition]
    templates: TemplatesConfig
    field_query_hints: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    question_groups: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    prompt_overrides: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("stages")
    @classmethod
    def validate_stages(cls, stages: List[StageBand]) -> List[StageBand]:
        if not stages:
            raise ValueError("At least one stage is required")
        seen = set()
        for stage in stages:
            if stage.id in seen:
                raise ValueError(f"Duplicate stage id: {stage.id}")
            seen.add(stage.id)
            if stage.max_headcount is not None and stage.max_headcount < stage.min_headcount:
                raise ValueError(
                    f"Stage {stage.id} has max_headcount smaller than min_headcount"
                )
        return stages

    @field_validator("checks")
    @classmethod
    def validate_checks(cls, checks: List[CheckDefinition]) -> List[CheckDefinition]:
        if len(checks) < 1:
            raise ValueError("At least one check is required")
        seen = set()
        for check in checks:
            if check.id in seen:
                raise ValueError(f"Duplicate check id: {check.id}")
            seen.add(check.id)
        return checks


class StageInferenceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage_id: str
    stage_label: str
    confidence: float = Field(ge=0, le=1)
    stage_min: Optional[str] = None
    stage_max: Optional[str] = None
    stage_point_estimate: Optional[str] = None
    stage_confidence: Optional[float] = Field(default=None, ge=0, le=1)
    stage_evidence: List[Citation] = Field(default_factory=list)
    explicit_headcount_evidence: bool = False
    drivers: List[str] = Field(default_factory=list)
    signals: List[str] = Field(default_factory=list)
    candidates: List["StageCandidate"] = Field(default_factory=list)
    note: Optional[str] = None
    source: Literal["rules", "unknown"] = "rules"


class StageCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage_id: str
    confidence: float = Field(ge=0, le=1)
    drivers: List[str] = Field(default_factory=list)
    signals: List[str] = Field(default_factory=list)
