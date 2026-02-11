from __future__ import annotations

import json
import os
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import List

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.ingest.loaders import parse_url_candidates
from app.logging_setup import setup_logging
from app.pipeline import run_assessment_pipeline
from app.report.exporters import REPORT_TITLE, markdown_to_docx_bytes, markdown_to_pdf_bytes

logger = logging.getLogger(__name__)


def _env_openai_timeout_seconds(name: str, default: int = 90) -> int | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if not value:
        return default
    if value in {"0", "none", "off", "false", "no", "inf", "infinite", "infinity"}:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return default
    if parsed <= 0:
        return None
    return parsed


def _parse_urls(raw_urls: str | None) -> List[str]:
    return parse_url_candidates(raw_urls)


class ExportPayload(BaseModel):
    markdown: str
    filename: str | None = "hr_assessment_report"


def _safe_filename(value: str | None, fallback: str) -> str:
    raw = (value or "").strip() or fallback
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw).strip("_")
    return cleaned or fallback


def _create_request_workspace(store_root: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    request_id = f"{ts}_{uuid.uuid4().hex[:8]}"
    request_dir = store_root / request_id
    (request_dir / "files").mkdir(parents=True, exist_ok=False)
    return request_dir


def build_app() -> FastAPI:
    setup_logging()
    app = FastAPI(title="HR Growth Assessment Web App", version="0.2.0")

    store_root = Path(os.getenv("HR_REPORT_INPUT_STORE", "data/input_requests"))
    store_root.mkdir(parents=True, exist_ok=True)

    default_profile = os.getenv("HR_REPORT_PROFILE", "tuning/profile.yaml")
    default_model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    timeout_seconds = _env_openai_timeout_seconds("OPENAI_TIMEOUT_SECONDS", default=90)
    default_output_root = os.getenv("HR_REPORT_OUTPUT_DIR", "out")
    progress_store: dict[str, dict] = {}
    progress_lock = Lock()

    static_root = Path(__file__).resolve().parent / "static"

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/config")
    async def config() -> dict:
        return {
            "profile": default_profile,
            "model": default_model,
            "timeout_seconds": timeout_seconds,
            "mode": "single_production_flow",
        }

    def _set_progress(
        token: str,
        *,
        state: str,
        stage: str,
        percent: float,
        message: str,
        error: str | None = None,
        result: dict | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        safe_percent = max(0.0, min(100.0, float(percent)))
        with progress_lock:
            entry = dict(progress_store.get(token, {}))
            entry.update(
                {
                    "token": token,
                    "state": state,
                    "stage": stage,
                    "percent": safe_percent,
                    "message": message,
                    "updated_at": now,
                }
            )
            entry.setdefault("started_at", now)
            if error:
                entry["error"] = error
            if result is not None:
                entry["result"] = result
            progress_store[token] = entry

    @app.get("/api/progress/{progress_token}")
    async def progress(progress_token: str) -> dict:
        with progress_lock:
            payload = progress_store.get(progress_token)
        if payload is not None:
            return payload
        now = datetime.now(timezone.utc).isoformat()
        # Frontend may poll before /api/analyze registers the token; return a non-error pending status.
        return {
            "token": progress_token,
            "state": "pending",
            "stage": "queued",
            "percent": 0.0,
            "message": "Waiting for analysis request.",
            "started_at": now,
            "updated_at": now,
        }

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return (static_root / "index.html").read_text(encoding="utf-8")

    @app.get("/app.js")
    async def app_js() -> Response:
        return Response(
            content=(static_root / "app.js").read_text(encoding="utf-8"),
            media_type="application/javascript",
        )

    @app.get("/styles.css")
    async def styles_css() -> Response:
        return Response(
            content=(static_root / "styles.css").read_text(encoding="utf-8"),
            media_type="text/css",
        )

    @app.post("/api/analyze")
    async def analyze(
        text: str = Form(default=""),
        urls: str = Form(default=""),
        files: List[UploadFile] = File(default=[]),
        progress_token: str = Form(default=""),
    ) -> JSONResponse:
        resolved_profile = default_profile
        resolved_model = default_model
        resolved_urls = _parse_urls(urls)
        resolved_progress_token = (progress_token or uuid.uuid4().hex).strip()

        if not text.strip() and not resolved_urls and not files:
            raise HTTPException(status_code=400, detail="Provide text, url(s), or file(s).")

        _set_progress(
            resolved_progress_token,
            state="running",
            stage="queued",
            percent=1,
            message="Queued for processing.",
        )

        request_dir = _create_request_workspace(store_root)
        files_dir = request_dir / "files"

        try:
            saved_files = []
            for upload in files:
                if not upload.filename:
                    continue
                target = files_dir / Path(upload.filename).name
                content = await upload.read()
                target.write_bytes(content)
                saved_files.append(str(target))

            (request_dir / "input_text.txt").write_text(text or "", encoding="utf-8")
            (request_dir / "input_urls.txt").write_text("\n".join(resolved_urls), encoding="utf-8")
            request_meta = {
                "profile": resolved_profile,
                "model": resolved_model,
                "saved_files": saved_files,
            }
            (request_dir / "request.json").write_text(
                json.dumps(request_meta, indent=2),
                encoding="utf-8",
            )

            logger.info(
                "web_request_received request_dir=%s text_chars=%s urls=%s files=%s",
                request_dir,
                len(text or ""),
                len(resolved_urls),
                len(saved_files),
            )

            input_path = str(files_dir) if saved_files else None

            def progress_callback(stage: str, percent: float, message: str) -> None:
                _set_progress(
                    resolved_progress_token,
                    state="running",
                    stage=stage,
                    percent=percent,
                    message=message,
                )

            result = await run_in_threadpool(
                run_assessment_pipeline,
                profile_path=resolved_profile,
                input_path=input_path,
                text=text,
                urls=resolved_urls,
                model=resolved_model,
                timeout_seconds=timeout_seconds,
                output_root=default_output_root,
                snapshot_file=None,
                progress_callback=progress_callback,
            )
            _set_progress(
                resolved_progress_token,
                state="completed",
                stage="complete",
                percent=100,
                message="Assessment complete.",
                result={"output_dir": result.get("output_dir")},
            )

            payload = {
                "ok": True,
                "request_dir": str(request_dir),
                "progress_token": resolved_progress_token,
                "result": result,
            }
            logger.info("web_request_complete request_dir=%s output_dir=%s", request_dir, result["output_dir"])
            return JSONResponse(payload)
        except Exception as exc:  # pragma: no cover - runtime safety
            _set_progress(
                resolved_progress_token,
                state="failed",
                stage="error",
                percent=100,
                message="Assessment failed.",
                error=str(exc),
            )
            logger.exception("web_request_failed request_dir=%s", request_dir)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/export/docx")
    async def export_docx(payload: ExportPayload = Body(...)) -> Response:
        markdown = payload.markdown.strip()
        if not markdown:
            raise HTTPException(status_code=400, detail="Markdown is required for DOCX export.")

        try:
            content = markdown_to_docx_bytes(markdown, title=REPORT_TITLE)
        except Exception as exc:  # pragma: no cover - runtime safety
            logger.exception("export_docx_failed")
            raise HTTPException(status_code=500, detail=f"DOCX export failed: {exc}") from exc

        filename = f"{_safe_filename(payload.filename, 'hr_assessment_report')}.docx"
        return Response(
            content=content,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.post("/api/export/pdf")
    async def export_pdf(payload: ExportPayload = Body(...)) -> Response:
        markdown = payload.markdown.strip()
        if not markdown:
            raise HTTPException(status_code=400, detail="Markdown is required for PDF export.")

        try:
            content = markdown_to_pdf_bytes(markdown, title=REPORT_TITLE)
        except Exception as exc:  # pragma: no cover - runtime safety
            logger.exception("export_pdf_failed")
            raise HTTPException(status_code=500, detail=f"PDF export failed: {exc}") from exc

        filename = f"{_safe_filename(payload.filename, 'hr_assessment_report')}.pdf"
        return Response(
            content=content,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    return app


app = build_app()
