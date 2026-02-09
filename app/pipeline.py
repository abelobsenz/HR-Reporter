from __future__ import annotations

import re
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import jsonschema

from app.config_loader import load_consultant_pack
from app.ingest.chunker import chunk_documents, get_last_chunk_stats
from app.ingest.loaders import get_last_ingestion_stats, load_documents
from app.llm.client import OpenAIResponsesClient
from app.llm.extract_snapshot import (
    deterministic_plan_from_findings,
    extract_snapshot,
    generate_plan,
)
from app.logic.checks_engine import run_checks
from app.logic.field_retrieval import run_field_targeted_retrieval
from app.logic.scoring import build_growth_drivers, rank_findings_and_adjust_confidence
from app.logic.stage_inference import infer_stage
from app.models import CompanyPeopleSnapshot, FinalReport
from app.report.renderer import render_markdown
from app.utils import flatten_checks_required_fields, load_json, write_json

logger = logging.getLogger(__name__)


def _load_snapshot_from_file(path: str) -> CompanyPeopleSnapshot:
    payload = load_json(path)
    return CompanyPeopleSnapshot.model_validate(payload)


def _load_report_schema() -> dict:
    return load_json(Path(__file__).resolve().parent / "schemas" / "report_schema.json")


def _collect_tracked_fields(pack_payload: dict) -> List[str]:
    fields = flatten_checks_required_fields(pack_payload.get("checks", []))
    for extra in [
        "company_name",
        "headcount",
        "headcount_range",
        "primary_locations",
        "current_priorities",
    ]:
        if extra not in fields:
            fields.append(extra)
    return fields


SEVERITY_SCORES = {"low": 1, "medium": 2, "high": 3, "critical": 4}


def _normalize_question(question: str) -> str:
    import re

    cleaned = re.sub(r"[^a-z0-9\\s]", " ", question.lower())
    cleaned = re.sub(r"\\s+", " ", cleaned).strip()
    # Light semantic dedupe via normalization of common prefixes.
    cleaned = cleaned.replace("can you confirm", "").replace("do you have", "").strip()
    return cleaned


def _rank_follow_up_questions(
    findings: List[object],
    *,
    check_group_map: Dict[str, str] | None = None,
    cap: int = 8,
) -> List[str]:
    question_map: Dict[str, Dict[str, object]] = {}
    seen_groups = set()
    check_group_map = check_group_map or {}
    for finding in findings:
        base_check_id = str(finding.check_id).split(":", 1)[0]
        group = check_group_map.get(base_check_id)
        if group and group in seen_groups:
            continue
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
        if group:
            seen_groups.add(group)

    ranked = sorted(
        question_map.values(),
        key=lambda e: (int(e["count"]) * int(e["severity"]), int(e["severity"]), int(e["count"])),
        reverse=True,
    )
    return [str(item["question"]) for item in ranked[:cap]]


def _build_unknown_lines(
    unknown_paths: List[str],
    *,
    retrieval_statuses: Dict[str, str] | None = None,
    cap: int = 12,
) -> List[str]:
    retrieval_statuses = retrieval_statuses or {}
    unknowns: List[str] = []
    for path in unknown_paths:
        status = retrieval_statuses.get(path)
        if status == "NOT_RETRIEVED":
            unknowns.append(f"Coverage gap (NOT_RETRIEVED): {path}")
        elif status == "NOT_FOUND_IN_RETRIEVED":
            unknowns.append(f"Searched retrieved corpus; not observed (NOT_FOUND_IN_RETRIEVED): {path}")
        elif status == "MENTIONED_AMBIGUOUS":
            unknowns.append(f"Mentioned but ambiguous (MENTIONED_AMBIGUOUS): {path}")
        else:
            unknowns.append(f"Not provided in sources: {path}")
    return unknowns[:cap]


def _reviewed_sources_from_documents(documents: List[object]) -> List[str]:
    seen = set()
    out: List[str] = []
    for doc in documents:
        source = doc.source
        if source in seen:
            continue
        seen.add(source)
        out.append(source)
    return out


def _build_coverage_note(reviewed_sources: List[str]) -> str:
    _ = reviewed_sources
    return "Sources reviewed were primarily onboarding/time-off pages; other domains may not be covered."


def _build_check_group_map(pack_payload: dict) -> Dict[str, str]:
    groups: Dict[str, str] = {}
    for check in pack_payload.get("checks", []):
        check_id = check.get("id")
        group = check.get("question_group")
        if check_id and group:
            groups[str(check_id)] = str(group)
    return groups


def _enforce_plan_constraints(plan: object, findings: List[object]) -> object:
    present_chunk_ids = {
        citation.chunk_id
        for finding in findings
        if finding.evidence_status == "present"
        for citation in finding.evidence
    }

    for bucket in [plan.days_30, plan.days_60, plan.days_90]:
        for action in bucket:
            evidence_chunk_ids = {e.chunk_id for e in action.evidence}
            has_present_evidence = bool(present_chunk_ids.intersection(evidence_chunk_ids))
            has_unknown_dependency = (not has_present_evidence) or ("not_found" in evidence_chunk_ids)

            if has_unknown_dependency and not action.action.startswith("Conditional: if "):
                action.action = f"Conditional: if prerequisite capability is not in place, {action.action}"

            # Avoid role hallucination.
            action.rationale = re.sub(
                r"\\b(People Team|People Ops|People Operations|HR lead|HR team)\\b",
                "Owner: TBD / assign",
                action.rationale,
                flags=re.IGNORECASE,
            )

    return plan


def run_assessment_pipeline(
    *,
    pack_path: str,
    input_path: str | Path | None,
    text: str | None,
    urls: Sequence[str] | None,
    model: str,
    timeout_seconds: int,
    output_root: str | Path,
    snapshot_file: str | None = None,
    no_llm_plan: bool = False,
) -> dict:
    logger.info("pipeline_start pack=%s model=%s", pack_path, model)
    pack = load_consultant_pack(pack_path)

    documents = load_documents(input_path=input_path, pasted_text=text, urls=urls)
    chunks, chunk_index = chunk_documents(documents)
    ingestion_stats = get_last_ingestion_stats()
    chunk_stats = get_last_chunk_stats()
    logger.info(
        "ingestion_done docs=%s urls_seeded=%s urls_fetched=%s failures=%s crawl_enabled=%s directory_crawl_ran=%s",
        len(documents),
        ingestion_stats.get("urls_seeded", 0),
        ingestion_stats.get("urls_fetched", 0),
        ingestion_stats.get("url_fetch_failures", 0),
        ingestion_stats.get("crawl_enabled", False),
        ingestion_stats.get("directory_crawl_ran", False),
    )
    logger.info(
        "chunking_done candidate=%s kept=%s dropped=%s short_but_salient_kept=%s",
        chunk_stats.get("candidate_chunks", 0),
        chunk_stats.get("kept_chunks", 0),
        chunk_stats.get("dropped_chunks", 0),
        chunk_stats.get("short_but_salient_kept", 0),
    )

    tracked_fields = _collect_tracked_fields(pack.model_dump(mode="python"))
    retrieval_result = run_field_targeted_retrieval(
        chunks=chunks,
        documents=documents,
        tracked_fields=tracked_fields,
        pack=pack,
    )
    logger.info(
        "retrieval_complete fields=%s explicit=%s not_retrieved=%s",
        len(tracked_fields),
        retrieval_result.get("coverage_summary", {}).get("fields_with_explicit", 0),
        retrieval_result.get("coverage_summary", {}).get("fields_not_retrieved", 0),
    )

    client = OpenAIResponsesClient(model=model, timeout_seconds=timeout_seconds)

    if snapshot_file:
        logger.info("snapshot_source=file path=%s", snapshot_file)
        snapshot = _load_snapshot_from_file(snapshot_file)
    else:
        if not client.is_enabled():
            raise RuntimeError(
                "OPENAI_API_KEY is required when snapshot_file is not provided."
            )
        logger.info(
            "snapshot_llm_start model=%s chunks=%s tracked_fields=%s",
            model,
            len(chunks),
            len(tracked_fields),
        )
        snapshot = extract_snapshot(
            client=client,
            chunks=chunks,
            tracked_fields=tracked_fields,
            prompt_overrides=pack.prompt_overrides,
            targeted_retrieval=retrieval_result,
        )
        logger.info("snapshot_llm_done")

    stage_result = infer_stage(
        snapshot=snapshot,
        pack=pack,
        chunks=chunks,
    )

    checks_stage_id = (
        stage_result.stage_min
        if (not stage_result.explicit_headcount_evidence and stage_result.stage_min)
        else stage_result.stage_id
    )
    stage_label_map = {stage.id: stage.label for stage in pack.stages}
    checks_stage_label = stage_label_map.get(checks_stage_id or stage_result.stage_id, stage_result.stage_label)

    findings, risks, unknown_paths = run_checks(
        snapshot=snapshot,
        pack=pack,
        stage_id=checks_stage_id or stage_result.stage_id,
        stage_label=checks_stage_label,
    )

    ranked_findings, confidence = rank_findings_and_adjust_confidence(
        findings=findings,
        pack=pack,
        stage_confidence=stage_result.confidence,
        unknown_count=len(unknown_paths),
        tracked_field_count=len(tracked_fields),
    )
    logger.info(
        "ranking_done findings=%s top=%s confidence=%.3f",
        len(ranked_findings),
        ranked_findings[0].check_id if ranked_findings else "none",
        confidence,
    )

    if no_llm_plan or not client.is_enabled():
        plan = deterministic_plan_from_findings(ranked_findings)
    else:
        try:
            plan = generate_plan(
                client=client,
                stage_label=stage_result.stage_label,
                snapshot=snapshot,
                findings=ranked_findings[:8],
                prompt_overrides=pack.prompt_overrides,
            )
        except Exception:
            plan = deterministic_plan_from_findings(ranked_findings)
    plan = _enforce_plan_constraints(plan, ranked_findings)

    pack_payload = pack.model_dump(mode="python")
    check_group_map = _build_check_group_map(pack_payload)
    follow_up_questions = _rank_follow_up_questions(
        ranked_findings,
        check_group_map=check_group_map,
        cap=8,
    )
    coverage_summary = {
        **retrieval_result.get("coverage_summary", {}),
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
    report = FinalReport(
        stage=stage_result.stage_label,
        confidence=confidence,
        drivers=build_growth_drivers(stage_result.drivers, ranked_findings),
        top_growth_areas=ranked_findings,
        risks=risks,
        plan_30_60_90=plan,
        follow_up_questions=follow_up_questions,
        unknowns=_build_unknown_lines(
            unknown_paths,
            retrieval_statuses=retrieval_result.get("field_statuses", {}),
            cap=12,
        ),
        reviewed_sources=_reviewed_sources_from_documents(documents),
        coverage_summary=coverage_summary,
        retrieval_summary={
            "field_statuses": retrieval_result.get("field_statuses", {}),
            "queries_by_field": retrieval_result.get("queries_by_field", {}),
        },
        stage_note=stage_result.note,
        coverage_note=_build_coverage_note(_reviewed_sources_from_documents(documents)),
        disclaimer=pack.disclaimer,
    )

    report_payload = report.model_dump(mode="json")
    jsonschema.validate(instance=report_payload, schema=_load_report_schema())

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    unique = uuid.uuid4().hex[:8]
    out_dir = Path(output_root) / f"{ts}_{unique}"
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "report.json"
    md_path = out_dir / "report.md"
    snapshot_path = out_dir / "snapshot.json"
    chunks_index_path = out_dir / "chunks_index.json"
    run_meta_path = out_dir / "run_meta.json"

    write_json(report_path, report_payload)
    write_json(snapshot_path, snapshot.model_dump(mode="json"))
    write_json(chunks_index_path, chunk_index)
    run_meta = {
        "ingestion": ingestion_stats,
        "chunking": chunk_stats,
        "retrieval": {
            "coverage_summary": retrieval_result.get("coverage_summary", {}),
            "field_statuses": retrieval_result.get("field_statuses", {}),
            "coverage_notes": retrieval_result.get("coverage_notes", {}),
            "top_candidate_counts": {
                field: len(cands)
                for field, cands in retrieval_result.get("field_candidates", {}).items()
            },
        },
        "llm": client.stats(),
    }
    write_json(run_meta_path, run_meta)

    markdown = render_markdown(report, pack)
    md_path.write_text(markdown, encoding="utf-8")

    return {
        "output_dir": str(out_dir),
        "files": {
            "report_json": str(report_path),
            "report_md": str(md_path),
            "snapshot_json": str(snapshot_path),
            "chunks_index_json": str(chunks_index_path),
            "run_meta_json": str(run_meta_path),
        },
        "report": report_payload,
        "report_markdown": markdown,
    }
