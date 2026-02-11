from __future__ import annotations

import jsonschema
import os
import re
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

from app.profile_loader import load_consultant_profile
from app.ingest.chunker import chunk_documents, get_last_chunk_stats
from app.ingest.loaders import get_last_ingestion_stats, load_documents
from app.llm.client import APICallTranscriptWriter, OpenAIResponsesClient
from app.llm.extract_snapshot import (
    extract_snapshot_from_evidence,
)
from app.logic.discovery import discover_additional_observations
from app.logic.evidence_collector import EvidenceCollector
from app.logic.evidence_memo import build_evidence_memo
from app.logic.profile_evaluator import evaluate_profile_expectations
from app.logic.scorecard import build_functional_scorecard
from app.logic.stage_guidance import build_stage_based_recommendation
from app.logic.scoring import build_growth_drivers, rank_findings
from app.logic.stage_inference import infer_stage
from app.models import AssessmentBundle, CompanyPeopleSnapshot, ConsultantProfile, FinalReport, SourceBundle
from app.report.citation_auditor import (
    audit_markdown_citations,
    sanitize_markdown_citations,
    sanitize_report_citations,
)
from app.report.renderer import render_markdown
from app.report.reviser import revise_markdown_report
from app.utils import load_json, write_json

logger = logging.getLogger(__name__)

SEVERITY_SCORES = {"low": 1, "medium": 2, "high": 3, "critical": 4}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _emit_progress(
    callback: Callable[[str, float, str], None] | None,
    *,
    stage: str,
    percent: float,
    message: str,
) -> None:
    if callback is None:
        return
    try:
        callback(stage, float(percent), message)
    except Exception:  # pragma: no cover - telemetry must not break analysis
        logger.exception("progress_callback_failed stage=%s percent=%s", stage, percent)


def _load_snapshot_from_file(path: str) -> CompanyPeopleSnapshot:
    payload = load_json(path)
    return CompanyPeopleSnapshot.model_validate(payload)


def _collect_tracked_fields_from_profile(profile_payload: dict) -> List[str]:
    fields: List[str] = []
    seen = set()
    for expectation in profile_payload.get("expectations", []):
        if not isinstance(expectation, dict):
            continue
        for field in expectation.get("snapshot_fields", []):
            if field in seen:
                continue
            seen.add(field)
            fields.append(field)

    for extra in ["company_name", "headcount", "headcount_range", "primary_locations", "current_priorities"]:
        if extra not in seen:
            seen.add(extra)
            fields.append(extra)
    return fields


def _normalize_question(question: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", question.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.replace("can you confirm", "").replace("do you have", "").strip()
    return cleaned


def _rank_follow_up_questions(findings: List[object], cap: int = 8) -> List[str]:
    question_map: Dict[str, Dict[str, object]] = {}
    for finding in findings:
        for question in finding.questions:
            key = _normalize_question(question)
            if not key:
                continue
            entry = question_map.setdefault(
                key,
                {
                    "question": question.strip(),
                    "count": 0,
                    "severity": 0,
                },
            )
            entry["count"] = int(entry["count"]) + 1
            entry["severity"] = max(int(entry["severity"]), SEVERITY_SCORES.get(finding.severity, 1))

    ranked = sorted(
        question_map.values(),
        key=lambda e: (int(e["count"]) * int(e["severity"]), int(e["severity"]), int(e["count"])),
        reverse=True,
    )
    return [str(item["question"]) for item in ranked[:cap]]


def _build_unknown_lines(unknown_paths: List[str], retrieval_statuses: Dict[str, str] | None = None) -> List[str]:
    retrieval_statuses = retrieval_statuses or {}
    out: List[str] = []
    for path in unknown_paths:
        status = retrieval_statuses.get(path)
        if status == "NOT_RETRIEVED":
            out.append(f"Coverage gap (NOT_RETRIEVED): {path}")
        elif status == "NOT_FOUND_IN_RETRIEVED":
            out.append(f"Searched retrieved corpus; not observed (NOT_FOUND_IN_RETRIEVED): {path}")
        elif status == "MENTIONED_AMBIGUOUS":
            out.append(f"Mentioned but ambiguous (MENTIONED_AMBIGUOUS): {path}")
        else:
            out.append(f"Not provided in sources: {path}")
    return out[:12]


def _reviewed_sources_from_documents(documents: List[object]) -> List[str]:
    seen = set()
    out: List[str] = []
    for doc in documents:
        if doc.source in seen:
            continue
        seen.add(doc.source)
        out.append(doc.source)
    return out


def _build_coverage_note(reviewed_sources: List[str]) -> str:
    _ = reviewed_sources
    return "Sources were prioritized toward broad HR controls; niche/local pages were deprioritized unless directly relevant."


def _stage_label_lookup(profile: ConsultantProfile) -> Dict[str, str]:
    out = {stage.id: stage.label for stage in profile.stages}
    for funding in profile.funding_stages:
        out[funding.id] = funding.label
    return out


def _stage_signal_label(signal: str) -> str:
    labels = {
        "explicit_headcount": "Explicit headcount statement",
        "headcount_range": "Headcount range statement",
        "funding_round": "Explicit funding-round signal",
        "primary_locations": "Primary location coverage",
    }
    return labels.get(signal, signal.replace("_", " "))


def _build_methodology_summary(*, documents: List[object], coverage_summary: Dict[str, object]) -> List[str]:
    return [
        "Reviewed uploaded files, pasted notes, and supplied URLs.",
        "Extracted evidence by HR control using deterministic retrieval over chunked sources.",
        (
            "Generated findings only when controls were explicitly missing or not confirmed in sources; "
            "explicit citations were retained where available."
        ),
        (
            f"Reviewed {int(coverage_summary.get('retrieved_docs', 0))} document(s) and "
            f"{int(coverage_summary.get('retrieved_chunks', 0))} evidence chunk(s)."
        ),
        f"Source count in assessment packet: {len(documents)}.",
    ]


def _build_data_limitations(
    *,
    unknown_lines: List[str],
    coverage_summary: Dict[str, object],
) -> List[str]:
    limitations: List[str] = []
    not_retrieved = int(coverage_summary.get("fields_not_retrieved", 0) or 0)
    not_found = int(coverage_summary.get("fields_not_found", 0) or 0)
    if not_retrieved > 0:
        limitations.append(
            f"{not_retrieved} tracked field(s) were not retrievable from the reviewed corpus and may require follow-up."
        )
    if not_found > 0:
        limitations.append(
            f"{not_found} tracked field(s) were searched but not explicitly found in retrieved sources."
        )
    limitations.extend(unknown_lines[:5])
    if not limitations:
        limitations.append("No major source coverage limitations were detected from ingestion metrics.")
    return limitations


def _derive_company_context(text: str | None, documents: List[object]) -> str:
    lines: List[str] = []
    if text and text.strip():
        lines.append("Input text excerpt:")
        lines.append(" ".join(text.strip().split())[:600])
    url_sources = [doc.source for doc in documents if getattr(doc, "source_type", "") == "url"]
    if url_sources:
        lines.append("Reviewed URL sources:")
        lines.append(", ".join(url_sources[:6]))
    return "\n".join(lines).strip()


def _compact_goal_text(value: str, max_chars: int = 220) -> str:
    text = " ".join((value or "").split()).strip()
    if not text:
        return "Generate an evidence-grounded HR assessment report for startup companies by size and funding stage."
    text = re.sub(r"\b(avoid|do not|don't|never)\b[^.]*[.]?", "", text, flags=re.IGNORECASE)
    text = " ".join(text.split()).strip()
    if not text:
        text = "Generate an evidence-grounded HR assessment report for startup companies by size and funding stage."
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars].rstrip()
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return f"{clipped.rstrip(' ,;:')}"


def _build_assessment_context(profile: ConsultantProfile) -> str:
    focus = _compact_goal_text(
        profile.assessment_focus or "Generate an evidence-grounded HR assessment report."
    )
    return (
        f"Goal: {focus}\n"
        "Task contribution: extract source-grounded evidence and synthesize stage-fit HR structure, "
        "practices, and risks."
    )


def run_assessment_pipeline(
    *,
    profile_path: str | None = None,
    input_path: str | Path | None,
    text: str | None,
    urls: Sequence[str] | None,
    model: str,
    timeout_seconds: int | None,
    output_root: str | Path,
    snapshot_file: str | None = None,
    progress_callback: Callable[[str, float, str], None] | None = None,
) -> dict:
    if profile_path is None:
        profile_path = "tuning/profile.yaml"
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    api_trace_root = Path(os.getenv("HR_REPORT_API_TRACE_DIR", "data/api_call_logs"))
    api_trace_path = api_trace_root / f"{run_id}.md"
    transcript_writer = APICallTranscriptWriter(
        path=api_trace_path,
        run_id=run_id,
        model=model,
        profile_path=profile_path,
        timeout_seconds=timeout_seconds,
    )
    logger.info("pipeline_start profile=%s model=%s", profile_path, model)
    _emit_progress(
        progress_callback,
        stage="ingest",
        percent=4,
        message="Loading sources and preparing ingestion.",
    )
    profile = load_consultant_profile(profile_path)

    bundle = AssessmentBundle(
        sources=SourceBundle(
            text=text or "",
            urls=list(urls or []),
            file_paths=([str(input_path)] if input_path else []),
        ),
        profile_id="default",
    )
    logger.debug(
        "assessment_bundle sources_text=%s urls=%s files=%s",
        bool(bundle.sources.text.strip()),
        len(bundle.sources.urls),
        len(bundle.sources.file_paths),
    )

    documents = load_documents(
        input_path=input_path,
        pasted_text=text,
        urls=urls,
        max_urls_per_domain=profile.retrieval_policy.max_urls_per_domain,
        max_total_urls=profile.retrieval_policy.max_total_urls,
        locale_bias=profile.retrieval_policy.locale_bias,
    )
    chunk_max_chars = max(800, int(os.getenv("HR_REPORT_CHUNK_MAX_CHARS", "5200")))
    chunk_overlap_chars = max(80, int(os.getenv("HR_REPORT_CHUNK_OVERLAP_CHARS", "480")))
    chunk_mode = (os.getenv("HR_REPORT_CHUNK_MODE", "legacy").strip().lower() or "legacy")
    _emit_progress(
        progress_callback,
        stage="chunk",
        percent=18,
        message="Chunking source documents.",
    )
    chunks, chunk_index = chunk_documents(
        documents,
        max_chars=chunk_max_chars,
        overlap_chars=chunk_overlap_chars,
        chunk_mode=chunk_mode,
    )
    ingestion_stats = get_last_ingestion_stats()
    chunk_stats = get_last_chunk_stats()

    logger.info(
        "ingestion_done docs=%s urls_seeded=%s urls_fetched=%s failures=%s",
        len(documents),
        ingestion_stats.get("urls_seeded", 0),
        ingestion_stats.get("urls_fetched", 0),
        ingestion_stats.get("url_fetch_failures", 0),
    )

    tracked_fields = _collect_tracked_fields_from_profile(profile.model_dump(mode="python"))
    assessment_context = _build_assessment_context(profile)
    client = OpenAIResponsesClient(
        model=model,
        timeout_seconds=timeout_seconds,
        transcript_writer=transcript_writer,
    )

    _emit_progress(
        progress_callback,
        stage="retrieve",
        percent=34,
        message="Collecting evidence from chunked sources.",
    )
    collector = EvidenceCollector(
        profile=profile,
        chunks=chunks,
        documents=documents,
        client=client,
    )
    evidence_result = collector.collect()

    _emit_progress(
        progress_callback,
        stage="snapshot",
        percent=50,
        message="Extracting company snapshot from evidence.",
    )
    snapshot_analysis_capture: Dict[str, Any] = {}
    if snapshot_file:
        snapshot = _load_snapshot_from_file(snapshot_file)
    else:
        snapshot = extract_snapshot_from_evidence(
            client=client,
            field_evidence=evidence_result.get("field_evidence", {}),
            tracked_fields=tracked_fields,
            retrieval_statuses=evidence_result.get("field_statuses", {}),
            retrieval_queries=evidence_result.get("field_queries", {}),
            prompt_overrides=profile.prompt_overrides,
            company_context=_derive_company_context(text, documents),
            assessment_context=assessment_context,
            source_chunks=chunks,
            analysis_capture=snapshot_analysis_capture,
        )

    stage_result = infer_stage(
        snapshot=snapshot,
        profile=profile,
        chunks=chunks,
    )
    analysis_rows = snapshot_analysis_capture.get("field_assessments", [])
    if not isinstance(analysis_rows, list):
        analysis_rows = []
    evidence_memo = build_evidence_memo(
        field_assessments=analysis_rows,
        retrieval_statuses=evidence_result.get("field_statuses", {}),
        retrieval_queries=evidence_result.get("field_queries", {}),
        stage_label=stage_result.company_stage_label or stage_result.stage_label,
        stage_drivers=stage_result.drivers,
        stage_note=stage_result.note,
    )

    _emit_progress(
        progress_callback,
        stage="evaluate",
        percent=64,
        message="Evaluating profile expectations.",
    )
    rubric_findings, rubric_risks, unknown_paths = evaluate_profile_expectations(
        snapshot=snapshot,
        profile=profile,
        stage_id=stage_result.stage_id,
        stage_label=stage_result.stage_label,
        evidence_result=evidence_result,
    )

    discovery_findings = discover_additional_observations(
        profile=profile,
        stage_label=stage_result.company_stage_label or stage_result.stage_label,
        evidence_result=evidence_result,
        client=client,
        assessment_context=assessment_context,
    )
    _emit_progress(
        progress_callback,
        stage="discovery",
        percent=74,
        message="Synthesizing additional observations.",
    )

    ranked_rubric = rank_findings(rubric_findings)
    ranked_discovery = rank_findings(discovery_findings)
    combined = rank_findings(ranked_rubric + ranked_discovery)
    coverage_summary = {
        **evidence_result.get("coverage_summary", {}),
        "urls_seeded": ingestion_stats.get("urls_seeded", 0),
        "urls_fetched": ingestion_stats.get("urls_fetched", 0),
        "url_fetch_failures": ingestion_stats.get("url_fetch_failures", 0),
        "url_redirects": ingestion_stats.get("url_redirects", 0),
        "domains": ingestion_stats.get("domains", []),
        "crawl_enabled": ingestion_stats.get("crawl_enabled", False),
        "directory_crawl_ran": ingestion_stats.get("directory_crawl_ran", False),
        "crawl_depth": ingestion_stats.get("crawl_depth", 1),
        "chunk_stats": chunk_stats,
    }
    scorecard = build_functional_scorecard(findings=combined, coverage_summary=coverage_summary)
    stage_guidance = build_stage_based_recommendation(
        profile=profile,
        size_stage_id=stage_result.stage_id,
        size_stage_label=stage_result.stage_label,
        funding_stage_id=stage_result.funding_stage_id,
        funding_stage_label=stage_result.funding_stage_label,
    )

    follow_up_questions = _rank_follow_up_questions(combined, cap=8)
    unknown_lines = _build_unknown_lines(unknown_paths, evidence_result.get("field_statuses", {}))
    stage_confidence = stage_result.stage_confidence if stage_result.stage_confidence is not None else stage_result.confidence
    provisional_threshold = float(os.getenv("HR_REPORT_STAGE_PROVISIONAL_THRESHOLD", "0.62"))
    stage_provisional = bool(stage_confidence is not None and stage_confidence < provisional_threshold)
    stage_lookup = _stage_label_lookup(profile)
    alternate_stage_candidates: List[str] = []
    for candidate in sorted(stage_result.candidates, key=lambda row: row.confidence, reverse=True):
        if candidate.stage_id == stage_result.stage_id:
            continue
        label = stage_lookup.get(candidate.stage_id, candidate.stage_id)
        alternate_stage_candidates.append(f"{label} ({candidate.confidence:.2f})")
        if len(alternate_stage_candidates) >= 2:
            break
    stage_signals_used = list(dict.fromkeys(stage_result.drivers + [f"Proxy signal: {signal}" for signal in stage_result.signals]))
    stage_signals_missing = [_stage_signal_label(signal) for signal in stage_result.signals_missing]
    stage_display_label = stage_result.company_stage_label or stage_result.stage_label
    if stage_provisional:
        stage_display_label = f"Provisional: {stage_display_label}"

    report = FinalReport(
        stage=stage_display_label,
        size_stage=stage_result.stage_label,
        funding_stage=stage_result.funding_stage_label,
        company_stage=stage_display_label,
        drivers=build_growth_drivers(stage_result.drivers, combined),
        stage_confidence=stage_confidence,
        stage_provisional=stage_provisional,
        signals_used=stage_signals_used,
        signals_missing=stage_signals_missing,
        alternate_stage_candidates=alternate_stage_candidates,
        functional_scorecard=scorecard,
        stage_based_recommendation=stage_guidance,
        profile_expectations=ranked_rubric,
        additional_observations=ranked_discovery,
        top_growth_areas=combined,
        risks=rubric_risks,
        methodology_summary=_build_methodology_summary(documents=documents, coverage_summary=coverage_summary),
        data_limitations=_build_data_limitations(unknown_lines=unknown_lines, coverage_summary=coverage_summary),
        follow_up_questions=follow_up_questions,
        unknowns=unknown_lines,
        reviewed_sources=_reviewed_sources_from_documents(documents),
        coverage_summary=coverage_summary,
        retrieval_summary={
            "field_statuses": evidence_result.get("field_statuses", {}),
            "field_queries": evidence_result.get("field_queries", {}),
            "expectation_statuses": evidence_result.get("expectation_statuses", {}),
            "expectation_queries": evidence_result.get("expectation_queries", {}),
        },
        stage_note=stage_result.note,
        coverage_note=_build_coverage_note(_reviewed_sources_from_documents(documents)),
    )

    report_payload = report.model_dump(mode="json")
    report_payload, citation_cleanup_summary = sanitize_report_citations(
        report_payload=report_payload,
        chunks_index=chunk_index,
    )
    report = FinalReport.model_validate(report_payload)
    jsonschema.validate(instance=report_payload, schema=FinalReport.model_json_schema())
    _emit_progress(
        progress_callback,
        stage="render",
        percent=95,
        message="Rendering report outputs.",
    )

    out_dir = Path(output_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "report.json"
    md_path = out_dir / "report.md"
    snapshot_path = out_dir / "snapshot.json"
    chunks_index_path = out_dir / "chunks_index.json"
    evidence_memo_path = out_dir / "evidence_memo.json"
    citation_audit_path = out_dir / "citation_audit.json"
    run_meta_path = out_dir / "run_meta.json"

    write_json(report_path, report_payload)
    write_json(snapshot_path, snapshot.model_dump(mode="json"))
    write_json(chunks_index_path, chunk_index)
    write_json(evidence_memo_path, evidence_memo)

    render_mode = "template"
    markdown = render_markdown(report, profile_name=profile.name)
    reviser_enabled = _env_bool("HR_REPORT_REPORT_REVISER_ENABLED", default=True)
    reviser_result: Dict[str, Any] = {
        "enabled": reviser_enabled,
        "attempted": False,
        "applied": False,
        "reason": "disabled_by_env",
    }
    if reviser_enabled:
        _emit_progress(
            progress_callback,
            stage="render",
            percent=97,
            message="Revising final report wording.",
        )
        markdown, reviser_result = revise_markdown_report(
            client=client,
            markdown=markdown,
            profile_name=profile.name,
        )

    markdown, markdown_cleanup_summary = sanitize_markdown_citations(
        markdown=markdown,
        chunks_index=chunk_index,
    )
    md_path.write_text(markdown, encoding="utf-8")
    citation_audit = audit_markdown_citations(
        markdown=markdown,
        chunks_index=chunk_index,
        report_payload=report_payload,
    )
    citation_audit["cleanup"] = {
        "report_citation_cleanup": citation_cleanup_summary,
        "markdown_citation_cleanup": markdown_cleanup_summary,
    }
    write_json(citation_audit_path, citation_audit)

    run_meta = {
        "ingestion": ingestion_stats,
        "chunking": chunk_stats,
        "evidence": {
            "coverage_summary": evidence_result.get("coverage_summary", {}),
            "field_statuses": evidence_result.get("field_statuses", {}),
            "expectation_statuses": evidence_result.get("expectation_statuses", {}),
            "top_candidate_counts": {
                field: len(rows)
                for field, rows in evidence_result.get("field_evidence", {}).items()
            },
            "snapshot_analysis": {
                "mode": snapshot_analysis_capture.get("analysis_mode"),
                "tracked_fields": snapshot_analysis_capture.get("tracked_fields", []),
                "field_assessment_count": len(analysis_rows),
                "resolver_model": snapshot_analysis_capture.get("resolver_model"),
            },
            "evidence_memo_summary": evidence_memo.get("summary", {}),
        },
        "report_rendering": {
            "mode": render_mode,
            "reviser": reviser_result,
            "citation_audit_passed": bool(citation_audit.get("passed", False)),
            "citation_audit_summary": citation_audit.get("summary", {}),
            "citation_cleanup": citation_audit.get("cleanup", {}),
        },
        "llm": client.stats(),
        "api_trace_md": str(api_trace_path),
    }
    write_json(run_meta_path, run_meta)
    _emit_progress(
        progress_callback,
        stage="complete",
        percent=100,
        message="Assessment complete.",
    )

    return {
        "output_dir": str(out_dir),
        "files": {
            "report_json": str(report_path),
            "report_md": str(md_path),
            "snapshot_json": str(snapshot_path),
            "chunks_index_json": str(chunks_index_path),
            "evidence_memo_json": str(evidence_memo_path),
            "citation_audit_json": str(citation_audit_path),
            "run_meta_json": str(run_meta_path),
            "api_trace_md": str(api_trace_path),
        },
        "report": report_payload,
        "report_markdown": markdown,
    }
