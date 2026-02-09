from __future__ import annotations

import json
import os
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response

from app.ingest.loaders import parse_url_candidates
from app.logging_setup import setup_logging
from app.pipeline import run_assessment_pipeline

logger = logging.getLogger(__name__)

def _parse_urls(raw_urls: str | None) -> List[str]:
    return parse_url_candidates(raw_urls)


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

    default_pack = os.getenv("HR_REPORT_PACK", "tuning/packs/default_pack.yaml")
    default_model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    timeout_seconds = int(os.getenv("OPENAI_TIMEOUT_SECONDS", "90"))
    default_output_root = os.getenv("HR_REPORT_OUTPUT_DIR", "out")
    default_no_llm_plan = os.getenv("HR_REPORT_NO_LLM_PLAN", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }

    static_root = Path(__file__).resolve().parent / "static"

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/config")
    async def config() -> dict:
        return {
            "pack": default_pack,
            "model": default_model,
            "timeout_seconds": timeout_seconds,
            "mode": "single_production_flow",
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
    ) -> JSONResponse:
        resolved_pack = default_pack
        resolved_model = default_model
        resolved_urls = _parse_urls(urls)
        use_no_llm_plan = default_no_llm_plan

        if not text.strip() and not resolved_urls and not files:
            raise HTTPException(status_code=400, detail="Provide text, url(s), or file(s).")

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
                "pack": resolved_pack,
                "model": resolved_model,
                "no_llm_plan": use_no_llm_plan,
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

            result = run_assessment_pipeline(
                pack_path=resolved_pack,
                input_path=input_path,
                text=text,
                urls=resolved_urls,
                model=resolved_model,
                timeout_seconds=timeout_seconds,
                output_root=default_output_root,
                snapshot_file=None,
                no_llm_plan=use_no_llm_plan,
            )

            payload = {
                "ok": True,
                "request_dir": str(request_dir),
                "result": result,
            }
            logger.info("web_request_complete request_dir=%s output_dir=%s", request_dir, result["output_dir"])
            return JSONResponse(payload)
        except Exception as exc:  # pragma: no cover - runtime safety
            logger.exception("web_request_failed request_dir=%s", request_dir)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = build_app()
