# HRReporter

Scale-aware HR assessment app for consultants. Submit pasted text, URLs, and files; get evidence-grounded rubric findings, stage/size guidance, and an HR assessment report aligned to template structure.

## What Changed (Startup Guidance + Report Alignment)
- Runtime is now **profile-first** using one tuning file: `tuning/profile.yaml`.
- Deprecated pack-era runtime paths were pruned; the main pipeline is profile-first only.
- Snapshot extraction is now **evidence-first** (not full corpus prompting).
- URL crawl prioritization is stricter to reduce irrelevant/niche pages.
- Stage inference now supports both **company size bands** and **funding stage signals**.
- Profile guidance supports stage-based HR structure recommendations, practices, and risks.
- CSV checklist inputs are supported as first-class ingestion sources.

## Repo Layout
- `app/main.py` CLI entry (`run`, `serve`)
- `app/web/server.py` FastAPI server
- `app/pipeline.py` shared pipeline
- `app/logic/evidence_collector.py` unified retrieval/evidence pass
- `app/logic/profile_evaluator.py` rubric findings lane
- `app/logic/discovery.py` additional observations lane
- `app/logic/stage_guidance.py` stage-based recommendation lane
- `app/logic/scorecard.py` maturity/impact scorecard lane
- `tuning/profile.yaml` primary tuning file
- `app/schemas/profile_schema.json` profile schema
- `data/input_requests/` saved web requests
- `out/` generated outputs

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[dev]'
```

## Environment
- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `OPENAI_TIMEOUT_SECONDS`
- `HR_REPORT_PROFILE` (default: `tuning/profile.yaml`)
- `HR_REPORT_OUTPUT_DIR` (default: `out`)
- `HR_REPORT_INPUT_STORE` (default: `data/input_requests`)
- Discovery synthesis controls:
  - `HR_REPORT_DISCOVERY_MAX_PROMPT_CHARS` (default: `16000`, above this discovery switches to catalog-batch mode)
  - `HR_REPORT_DISCOVERY_CATALOG_BATCH_SIZE` (default: `28`)
- Snapshot evidence controls:
  - `HR_REPORT_SNAPSHOT_EVIDENCE_MAX_PROMPT_CHARS` (default: `24000`, above this the extractor switches to field-batch mode)
  - `HR_REPORT_SNAPSHOT_EVIDENCE_FIELD_BATCH_SIZE` (default: `12`)
- Chunking controls:
  - `HR_REPORT_CHUNK_MAX_CHARS` (default: `3200`)
  - `HR_REPORT_CHUNK_OVERLAP_CHARS` (default: `320`)
  - `HR_REPORT_CHUNK_MODE` (default: `legacy`; set `semantic` to enable heading-aware file chunking + tighter list/table chunks)
- URL controls:
  - `HR_REPORT_URL_TIMEOUT_SECONDS`
  - `HR_REPORT_URL_CRAWL_LIMIT`
  - `HR_REPORT_URL_MAX_TOTAL_FETCHES`
  - `HR_REPORT_URL_LOCALE_BIAS`

## Run Web App
```bash
python -m app.main serve --host 127.0.0.1 --port 8080
```
Open `http://127.0.0.1:8080`.

## Run CLI
```bash
python -m app.main run --profile tuning/profile.yaml --input inputs/company_pack/
```

## API
`POST /api/analyze` (multipart form)
- `text`
- `urls`
- `files[]`

Response includes:
- `request_dir`
- `result.output_dir`
- `result.report`
- `result.report_markdown`

Generated report sections include:
- Executive Summary
- Methodology and Data Sources
- HR Functional Scorecard
- Top Risks and Risk Flags
- Stage-Based Recommendations
- Functional Area Deep-Dives

Export endpoints:
- `POST /api/export/pdf` (JSON body: `markdown`, optional `filename`)
- `POST /api/export/docx` (JSON body: `markdown`, optional `filename`)

The web UI includes one-click buttons for:
- `Download PDF`
- `Download Word (.docx)`

## Output Artifacts
Each run still writes:
- `report.json`
- `report.md`
- `snapshot.json`
- `chunks_index.json`
- `run_meta.json`

## Evidence Semantics
Findings keep explicit evidence semantics:
- `present`
- `explicitly_missing`
- `not_provided_in_sources`
- `not_assessed`

Retrieval statuses remain explicit (`MENTIONED_EXPLICIT`, `MENTIONED_IMPLICIT`, etc.) for auditability.
