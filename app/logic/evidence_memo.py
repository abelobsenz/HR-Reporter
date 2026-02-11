from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List


def _compact_citations(citations: List[Dict[str, Any]], limit: int = 2) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in citations[:limit]:
        if not isinstance(row, dict):
            continue
        chunk_id = str(row.get("chunk_id", "")).strip()
        snippet = str(row.get("snippet", "")).strip()
        if not chunk_id or not snippet:
            continue
        out.append({"chunk_id": chunk_id, "snippet": snippet})
    return out


def _note_for_field(row: Dict[str, Any]) -> str:
    field_path = str(row.get("field_path", "")).strip()
    reasoning = str(row.get("hr_reasoning", "")).strip()
    value_text = str(row.get("value_text", "")).strip()
    confidence = float(row.get("confidence", 0.0) or 0.0)
    if value_text:
        return f"{field_path}: {value_text} (confidence={confidence:.2f})"
    if reasoning:
        return f"{field_path}: {reasoning} (confidence={confidence:.2f})"
    return f"{field_path}: unresolved (confidence={confidence:.2f})"


def build_evidence_memo(
    *,
    field_assessments: List[Dict[str, Any]],
    retrieval_statuses: Dict[str, str],
    retrieval_queries: Dict[str, List[str]],
    stage_label: str,
    stage_drivers: List[str],
    stage_note: str | None = None,
) -> Dict[str, Any]:
    confirmed: List[Dict[str, Any]] = []
    weak: List[Dict[str, Any]] = []
    conflicts: List[Dict[str, Any]] = []
    gaps: List[Dict[str, Any]] = []
    follow_ups: List[str] = []

    for row in field_assessments:
        if not isinstance(row, dict):
            continue
        field_path = str(row.get("field_path", "")).strip()
        if not field_path:
            continue
        verdict = str(row.get("verdict", "not_found")).strip()
        retrieval_status = str(retrieval_statuses.get(field_path, row.get("retrieval_status") or "NOT_RETRIEVED"))
        confidence = float(row.get("confidence", 0.0) or 0.0)
        citations = _compact_citations(row.get("citations", []))
        entry = {
            "field_path": field_path,
            "verdict": verdict,
            "retrieval_status": retrieval_status,
            "confidence": round(confidence, 3),
            "value_text": str(row.get("value_text", "")).strip(),
            "hr_reasoning": str(row.get("hr_reasoning", "")).strip(),
            "citations": citations,
        }
        if verdict == "present":
            confirmed.append(entry)
        elif verdict == "conflict":
            conflicts.append(entry)
        elif verdict in {"ambiguous", "not_found"}:
            gaps.append(entry)
            weak.append(entry)
        elif verdict == "explicitly_missing":
            weak.append(entry)

        if verdict in {"ambiguous", "conflict", "not_found"}:
            queries = retrieval_queries.get(field_path, [])[:2]
            if queries:
                question = f"Confirm {field_path} using direct source language (queries: {', '.join(queries)})."
            else:
                question = f"Confirm {field_path} with a direct policy/process source citation."
            if question not in follow_ups:
                follow_ups.append(question)

    summary = {
        "confirmed_fields": len(confirmed),
        "weak_fields": len(weak),
        "conflict_fields": len(conflicts),
        "coverage_gaps": len(gaps),
    }

    stage_fit_notes = [_note_for_field(row) for row in confirmed[:4]]
    if stage_drivers:
        stage_fit_notes.extend(stage_drivers[:4])
    if stage_note:
        stage_fit_notes.append(stage_note)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "stage_context": {"stage_label": stage_label, "stage_note": stage_note},
        "confirmed_signals": confirmed,
        "weak_signals": weak,
        "conflicts": conflicts,
        "coverage_gaps": gaps,
        "stage_fit_notes": stage_fit_notes[:12],
        "follow_up_questions": follow_ups[:12],
    }
