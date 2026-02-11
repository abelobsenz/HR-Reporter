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
        "ambiguous",
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
        "ambiguous",
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

        if self.evidence_status == "ambiguous":
            if len(all_citations) < 1:
                raise ValueError("ambiguous evidence_status requires at least one citation")

        if self.evidence_status in {"not_provided_in_sources", "not_assessed"} and len(all_citations) != 0:
            raise ValueError("not_provided_in_sources/not_assessed cannot include citations")

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
        "ambiguous",
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

        if self.evidence_status == "ambiguous" and len(self.citations) < 1:
            raise ValueError("ambiguous subcheck requires citations")

        if self.evidence_status in {"not_provided_in_sources", "not_assessed"} and len(self.citations) != 0:
            raise ValueError("not_provided_in_sources/not_assessed subcheck cannot include citations")

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


class GuidanceSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    url: str
    published_date: Optional[str] = None
    notes: Optional[str] = None


class GuidanceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    url: str
    published_date: Optional[str] = None


class FunctionalScorecardItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    functional_area: str
    maturity_level: Literal["Under-Developed", "Developing", "Aligned", "Strategic"]
    impact_level: Literal["Limited", "Moderate", "Significant"]
    rationale: Optional[str] = None


class StageBasedRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    size_stage: str
    funding_stage: Optional[str] = None
    hr_structure_recommendation: str
    recommended_practices: List[str] = Field(default_factory=list)
    potential_risks: List[str] = Field(default_factory=list)
    sources: List[GuidanceReference] = Field(default_factory=list)


class FinalReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: str
    size_stage: Optional[str] = None
    funding_stage: Optional[str] = None
    company_stage: Optional[str] = None
    drivers: List[str] = Field(default_factory=list)
    stage_confidence: Optional[float] = Field(default=None, ge=0, le=1)
    stage_provisional: bool = False
    signals_used: List[str] = Field(default_factory=list)
    signals_missing: List[str] = Field(default_factory=list)
    alternate_stage_candidates: List[str] = Field(default_factory=list)
    functional_scorecard: List[FunctionalScorecardItem] = Field(default_factory=list)
    stage_based_recommendation: Optional[StageBasedRecommendation] = None
    profile_expectations: List[Finding] = Field(default_factory=list)
    additional_observations: List[Finding] = Field(default_factory=list)
    top_growth_areas: List[Finding] = Field(default_factory=list)
    risks: List[RiskItem] = Field(default_factory=list)
    methodology_summary: List[str] = Field(default_factory=list)
    data_limitations: List[str] = Field(default_factory=list)
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


class SourceBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = ""
    urls: List[str] = Field(default_factory=list)
    file_paths: List[str] = Field(default_factory=list)


class AssessmentBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sources: SourceBundle
    profile_id: str = "default"
    company_context: Optional[str] = None


class ProfileExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    area: str
    claim: str
    severity_if_missing: Literal["low", "medium", "high", "critical"]
    evidence_queries: List[str] = Field(default_factory=list)
    snapshot_fields: List[str] = Field(default_factory=list)
    follow_up_question: Optional[str] = None
    compliance: bool = False


class ProfileStage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    min_headcount: int = Field(ge=0)
    max_headcount: Optional[int] = Field(default=None, ge=0)
    expectations: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_bounds(self) -> "ProfileStage":
        if self.max_headcount is not None and self.max_headcount < self.min_headcount:
            raise ValueError("max_headcount cannot be less than min_headcount")
        return self


class ProfileFundingStage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    aliases: List[str] = Field(default_factory=list)


class StartupGuidanceRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    size_stage_ids: List[str] = Field(default_factory=list)
    funding_stage_ids: List[str] = Field(default_factory=list)
    hr_structure_recommendation: str
    recommended_practices: List[str] = Field(default_factory=list)
    potential_risks: List[str] = Field(default_factory=list)
    source_ids: List[str] = Field(default_factory=list)


class RedFlagRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    area: str
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    pattern: str
    question: Optional[str] = None
    regex: bool = False


class RetrievalPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_urls_per_domain: int = Field(default=8, ge=1, le=30)
    max_total_urls: int = Field(default=60, ge=1, le=250)
    max_snippets_per_item: int = Field(default=6, ge=1, le=24)
    snippet_max_chars: int = Field(default=900, ge=120, le=2400)
    locale_bias: str = "us"


class DiscoveryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    max_findings: int = Field(default=8, ge=1, le=30)


class ConsultantProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    name: str
    assessment_focus: Optional[str] = None
    report_focus_points: List[str] = Field(default_factory=list)
    ask_questions_when_unknown: bool = True
    stages: List[ProfileStage]
    funding_stages: List[ProfileFundingStage] = Field(default_factory=list)
    startup_guidance_sources: List[GuidanceSource] = Field(default_factory=list)
    startup_guidance_rules: List[StartupGuidanceRule] = Field(default_factory=list)
    expectations: List[ProfileExpectation]
    red_flags: List[RedFlagRule] = Field(default_factory=list)
    retrieval_policy: RetrievalPolicy = Field(default_factory=RetrievalPolicy)
    discovery: DiscoveryPolicy = Field(default_factory=DiscoveryPolicy)
    prompt_overrides: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("stages")
    @classmethod
    def validate_profile_stages(cls, stages: List[ProfileStage]) -> List[ProfileStage]:
        if not stages:
            raise ValueError("At least one profile stage is required")
        seen = set()
        for stage in stages:
            if stage.id in seen:
                raise ValueError(f"Duplicate profile stage id: {stage.id}")
            seen.add(stage.id)
        return stages

    @field_validator("expectations")
    @classmethod
    def validate_expectations(cls, expectations: List[ProfileExpectation]) -> List[ProfileExpectation]:
        if not expectations:
            raise ValueError("At least one profile expectation is required")
        seen = set()
        for expectation in expectations:
            if expectation.id in seen:
                raise ValueError(f"Duplicate profile expectation id: {expectation.id}")
            seen.add(expectation.id)
        return expectations

    @field_validator("funding_stages")
    @classmethod
    def validate_funding_stages(cls, funding_stages: List[ProfileFundingStage]) -> List[ProfileFundingStage]:
        seen = set()
        for stage in funding_stages:
            if stage.id in seen:
                raise ValueError(f"Duplicate funding stage id: {stage.id}")
            seen.add(stage.id)
        return funding_stages

    @model_validator(mode="after")
    def validate_stage_expectation_links(self) -> "ConsultantProfile":
        expectation_ids = {expectation.id for expectation in self.expectations}
        for stage in self.stages:
            for expectation_id in stage.expectations:
                if expectation_id not in expectation_ids:
                    raise ValueError(
                        f"Stage '{stage.id}' references unknown expectation '{expectation_id}'"
                    )
        stage_ids = {stage.id for stage in self.stages}
        funding_stage_ids = {stage.id for stage in self.funding_stages}
        source_ids = {source.id for source in self.startup_guidance_sources}
        for rule in self.startup_guidance_rules:
            for stage_id in rule.size_stage_ids:
                if stage_id not in stage_ids:
                    raise ValueError(
                        f"Startup guidance rule '{rule.id}' references unknown size stage '{stage_id}'"
                    )
            for stage_id in rule.funding_stage_ids:
                if stage_id not in funding_stage_ids:
                    raise ValueError(
                        f"Startup guidance rule '{rule.id}' references unknown funding stage '{stage_id}'"
                    )
            for source_id in rule.source_ids:
                if source_id not in source_ids:
                    raise ValueError(
                        f"Startup guidance rule '{rule.id}' references unknown source '{source_id}'"
                    )
        return self


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
    funding_stage_id: Optional[str] = None
    funding_stage_label: Optional[str] = None
    funding_stage_confidence: Optional[float] = Field(default=None, ge=0, le=1)
    funding_stage_evidence: List[Citation] = Field(default_factory=list)
    company_stage_label: Optional[str] = None
    explicit_headcount_evidence: bool = False
    drivers: List[str] = Field(default_factory=list)
    signals: List[str] = Field(default_factory=list)
    signals_missing: List[str] = Field(default_factory=list)
    candidates: List["StageCandidate"] = Field(default_factory=list)
    note: Optional[str] = None
    source: Literal["rules", "unknown"] = "rules"


class StageCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage_id: str
    confidence: float = Field(ge=0, le=1)
    drivers: List[str] = Field(default_factory=list)
    signals: List[str] = Field(default_factory=list)
