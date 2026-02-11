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


def _line_mode(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return "blank"
    if _is_heading(stripped):
        return "heading"
    if re.match(r"^([-*]|\d+[.)])\s+", stripped):
        return "list"
    if "|" in stripped and len(stripped) >= 6:
        return "table"
    return "text"


def _split_blocks(text: str) -> List[str]:
    lines = text.splitlines()
    blocks: List[List[str]] = []
    current: List[str] = []
    current_mode = "text"

    def _flush() -> None:
        nonlocal current
        if not current:
            return
        block = "\n".join(current).strip()
        if block:
            blocks.append(current[:])
        current = []

    for line in lines:
        mode = _line_mode(line)
        if mode == "blank":
            _flush()
            current_mode = "text"
            continue
        if not current:
            current = [line]
            current_mode = mode
            continue
        # Keep coherent list/table runs intact, and let text paragraphs accumulate.
        if current_mode == mode and mode in {"list", "table", "text"}:
            current.append(line)
            continue
        # Headings should stand alone to avoid injecting formatting noise into neighboring blocks.
        if mode == "heading" or current_mode == "heading":
            _flush()
            current = [line]
            current_mode = mode
            continue
        # Merge short list/text transitions that are likely the same paragraph context.
        if {current_mode, mode} <= {"text", "list"} and len(" ".join(current).split()) < 28:
            current.append(line)
            current_mode = "text"
            continue
        _flush()
        current = [line]
        current_mode = mode
    _flush()

    out: List[str] = []
    for block_lines in blocks:
        block = "\n".join(block_lines).strip()
        if block:
            out.append(block)
    return out


def _split_long_block(block: str, max_chars: int, overlap_chars: int) -> List[str]:
    if len(block) <= max_chars:
        return [block]

    # Prefer sentence boundaries to avoid clipped evidence snippets.
    sentences = [
        segment.strip()
        for segment in re.split(r"(?<=[.!?])\s+", block)
        if segment.strip()
    ]
    if len(sentences) <= 1:
        sentences = [part.strip() for part in block.split("\n") if part.strip()]
    if len(sentences) <= 1:
        sentences = [block]

    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(sentence) <= max_chars:
            current = sentence
            continue
        # Last fallback for a single long sentence/line.
        start = 0
        while start < len(sentence):
            end = min(start + max_chars, len(sentence))
            if end < len(sentence):
                boundary = sentence.rfind(" ", start + int(max_chars * 0.6), end)
                if boundary > start:
                    end = boundary
            piece = sentence[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= len(sentence):
                break
            next_start = max(0, end - overlap_chars)
            if next_start <= start:
                next_start = end
            start = next_start
        current = ""
    if current:
        chunks.append(current)
    return chunks


def _chunk_text(text: str, max_chars: int = 3200, overlap_chars: int = 320) -> List[str]:
    stripped_text = text.strip()
    if len(stripped_text) <= max_chars:
        return [stripped_text]

    blocks = _split_blocks(stripped_text)
    if not blocks:
        blocks = [stripped_text]

    chunks: List[str] = []
    current = ""
    sep = "\n\n"

    for block in blocks:
        candidate = f"{current}{sep}{block}".strip() if current else block
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)

        if len(block) <= max_chars:
            current = block
            continue

        for part in _split_long_block(block, max_chars=max_chars, overlap_chars=overlap_chars):
            if len(part) <= max_chars:
                chunks.append(part)
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
    max_chars: int = 3200,
    overlap_chars: int = 320,
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
    max_chars: int = 3200,
    overlap_chars: int = 320,
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
        source_type = getattr(doc, "source_type", "text")
        apply_quality_gate = source_type == "url"
        split_by_headings = source_type == "url"
        logger.debug(
            "chunker_doc doc_id=%s source=%s source_type=%s text_chars=%s",
            doc.doc_id,
            doc.source,
            source_type,
            len(doc.text),
        )
        if split_by_headings:
            sections = _split_sections(doc.text)
            if not sections:
                sections = [("General", doc.text)]
        else:
            # File/text inputs usually carry dense relevant context; keep larger contiguous chunks.
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
                stripped_part = part.strip()
                short_or_sparse = (len(stripped_part) < 300) or (_sentence_count(stripped_part) < 2)
                salient = _is_salient(stripped_part)
                if apply_quality_gate and not _passes_quality_gate(stripped_part):
                    stats["dropped_chunks"] = int(stats["dropped_chunks"]) + 1
                    stats["dropped_low_info"] = int(stats["dropped_low_info"]) + 1
                    continue
                if apply_quality_gate and short_or_sparse and salient:
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
