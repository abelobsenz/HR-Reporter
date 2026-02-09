from __future__ import annotations

import re
import logging
from typing import Any, Dict, List, Tuple

from app.models import RawDocument, TextChunk

logger = logging.getLogger(__name__)

HEADING_PATTERNS = [
    re.compile(r"^\s{0,3}#{1,6}\s+(.+)$"),
    re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(.+)$"),
    re.compile(r"^\s*([A-Z][A-Z\s\-/]{3,80})\s*$"),
]
SALIENT_KEYWORDS = [
    "harassment",
    "complaint",
    "reporting",
    "retaliation",
    "hotline",
    "overtime",
    "exempt",
    "non-exempt",
    "jurisdiction",
    "required training",
    "eeo",
    "discrimination",
    "whistleblower",
    "eligibility",
    "benefits",
    "leave type",
    "leave policy",
]
_CHUNK_STATS: Dict[str, object] = {}


def _is_heading(line: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None
    for pattern in HEADING_PATTERNS:
        match = pattern.match(stripped)
        if match:
            return match.groups()[-1].strip()
    if stripped.endswith(":") and len(stripped.split()) <= 12:
        return stripped[:-1].strip()
    return None


def _split_sections(text: str) -> List[Tuple[str, str]]:
    lines = text.splitlines()
    sections: List[Tuple[str, List[str]]] = [("General", [])]

    for line in lines:
        heading = _is_heading(line)
        if heading:
            sections.append((heading, []))
            continue
        sections[-1][1].append(line)

    out: List[Tuple[str, str]] = []
    for section, content_lines in sections:
        section_text = "\n".join(content_lines).strip()
        if section_text:
            out.append((section, section_text))
    return out


def _chunk_text(text: str, max_chars: int = 1800, overlap_chars: int = 180) -> List[str]:
    if len(text) <= max_chars:
        return [text]

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)

        if len(paragraph) <= max_chars:
            current = paragraph
            continue

        # Fallback: hard-split very long paragraphs.
        start = 0
        while start < len(paragraph):
            end = min(start + max_chars, len(paragraph))
            if end < len(paragraph):
                # Prefer a whitespace boundary to avoid cutting words/snippets mid-token.
                boundary = paragraph.rfind(" ", start + int(max_chars * 0.6), end)
                if boundary > start:
                    end = boundary
            slice_text = paragraph[start:end].strip()
            if slice_text:
                chunks.append(slice_text)
            if end == len(paragraph):
                break
            next_start = max(0, end - overlap_chars)
            if next_start < len(paragraph):
                forward_space = paragraph.find(" ", next_start, min(len(paragraph), next_start + 80))
                if forward_space != -1:
                    next_start = forward_space + 1
            if next_start <= start:
                next_start = end
            start = next_start
        current = ""

    if current:
        chunks.append(current)

    return chunks


def _sentence_count(text: str) -> int:
    return len(re.findall(r"[.!?](?:\s|$)", text))


def _nav_score(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    short_lines = sum(1 for line in lines if len(line.split()) <= 4)
    punct_lines = sum(1 for line in lines if any(token in line for token in ".:;!?"))
    short_ratio = short_lines / len(lines)
    punct_ratio = punct_lines / len(lines)
    return round(short_ratio - punct_ratio, 3)


def _is_list_chunk(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    list_lines = sum(1 for line in lines if re.match(r"^([-*]|\d+[.)])\s+", line))
    return list_lines >= 2


def _is_table_chunk(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    pipe_lines = sum(1 for line in lines if "|" in line)
    return pipe_lines >= 2


def _is_salient(text: str) -> bool:
    lower = text.lower()
    return any(keyword in lower for keyword in SALIENT_KEYWORDS)


def _passes_quality_gate(text: str) -> bool:
    stripped = text.strip()
    low_length = len(stripped) < 300
    low_sentence_density = _sentence_count(stripped) < 2
    low_information = low_length or low_sentence_density
    return not (low_information and not _is_salient(stripped))


def _chunk_text_with_offsets(
    text: str,
    *,
    base_offset: int,
    max_chars: int = 1800,
    overlap_chars: int = 180,
) -> List[Tuple[str, int, int]]:
    parts = _chunk_text(text, max_chars=max_chars, overlap_chars=overlap_chars)
    out: List[Tuple[str, int, int]] = []
    cursor = 0
    for part in parts:
        if not part:
            continue
        idx = text.find(part, cursor)
        if idx < 0:
            idx = cursor
        start = base_offset + idx
        end = start + len(part)
        out.append((part, start, end))
        cursor = max(idx + len(part) - overlap_chars, idx + 1)
    return out


def chunk_documents(
    documents: List[RawDocument],
    max_chars: int = 1800,
    overlap_chars: int = 180,
) -> tuple[List[TextChunk], Dict[str, Dict[str, Any]]]:
    chunks: List[TextChunk] = []
    index: Dict[str, Dict[str, Any]] = {}
    stats = {
        "candidate_chunks": 0,
        "kept_chunks": 0,
        "dropped_chunks": 0,
        "dropped_low_info": 0,
        "short_but_salient_kept": 0,
    }

    logger.info(
        "chunker_start docs=%s max_chars=%s overlap=%s",
        len(documents),
        max_chars,
        overlap_chars,
    )
    for doc in documents:
        logger.debug("chunker_doc doc_id=%s source=%s text_chars=%s", doc.doc_id, doc.source, len(doc.text))
        sections = _split_sections(doc.text)
        if not sections:
            sections = [("General", doc.text)]

        seq = 1
        section_cursor = 0
        for section, text in sections:
            section_start = doc.text.find(text, section_cursor)
            if section_start < 0:
                section_start = section_cursor
            section_cursor = max(section_start + len(text), section_cursor)

            for part, start_char, end_char in _chunk_text_with_offsets(
                text,
                base_offset=section_start,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            ):
                stats["candidate_chunks"] = int(stats["candidate_chunks"]) + 1
                short_or_sparse = (len(part.strip()) < 300) or (_sentence_count(part.strip()) < 2)
                salient = _is_salient(part.strip())
                if not _passes_quality_gate(part):
                    stats["dropped_chunks"] = int(stats["dropped_chunks"]) + 1
                    stats["dropped_low_info"] = int(stats["dropped_low_info"]) + 1
                    continue
                if short_or_sparse and salient:
                    stats["short_but_salient_kept"] = int(stats["short_but_salient_kept"]) + 1
                chunk_id = f"{doc.doc_id}-c{seq:03d}"
                seq += 1
                chunk = TextChunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    section=section,
                    heading_path=[section],
                    is_list=_is_list_chunk(part),
                    is_table=_is_table_chunk(part),
                    start_char=start_char,
                    end_char=end_char,
                    nav_score=_nav_score(part),
                    text=part,
                )
                chunks.append(chunk)
                index[chunk_id] = {
                    "doc_id": doc.doc_id,
                    "section": section,
                    "heading_path": [section],
                    "is_list": chunk.is_list,
                    "is_table": chunk.is_table,
                    "start_char": start_char,
                    "end_char": end_char,
                    "nav_score": chunk.nav_score,
                    "text": part,
                    "source": doc.source,
                }
                stats["kept_chunks"] = int(stats["kept_chunks"]) + 1

    global _CHUNK_STATS
    _CHUNK_STATS = stats
    logger.info(
        "chunker_done candidate=%s kept=%s dropped=%s short_but_salient_kept=%s",
        stats["candidate_chunks"],
        stats["kept_chunks"],
        stats["dropped_chunks"],
        stats["short_but_salient_kept"],
    )

    return chunks, index


def get_last_chunk_stats() -> Dict[str, object]:
    return dict(_CHUNK_STATS)
