# HR Growth-Areas Assessment Tool

Web-first HR assessment app for solo consultants.

## What Changed
- Input requests are now designed to run through a web server on a port.
- Works in both environments:
  - local (`127.0.0.1:<port>`)
  - AWS/Lightsail (`0.0.0.0:<port>` behind your public IP)
- Request inputs are persisted in one folder: `data/input_requests/`.
- Tuning/setup assets are centralized in `tuning/`.

## Repository Layout
- `app/main.py`
  CLI entrypoint (`run` and `serve`)
- `app/pipeline.py`
  Shared analysis pipeline used by CLI and web
- `app/web/server.py`
  FastAPI app + API endpoints
- `app/web/static/`
  UI (drag/drop/paste/url-drop)
- `data/input_requests/`
  Saved request inputs (files/text/urls/request metadata)
- `tuning/packs/`
  Consultant packs used for stage/check/rubric behavior
- `tuning/setup/`
  Prompting/tuning reference
- `app/schemas/`
  JSON schemas for snapshot/report/pack
- `out/`
  Generated outputs

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[dev]'
```

## Environment Variables
The app auto-loads `.env`.

- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `OPENAI_TIMEOUT_SECONDS`
- `HR_REPORT_LOG_LEVEL` (default: `INFO`; set `DEBUG` for detailed crawl/retrieval/stage/check logs)
- `HR_REPORT_SNAPSHOT_MAX_CHUNKS` (default: `80`, pre-filter cap before snapshot extraction)
- `HR_REPORT_SNAPSHOT_MAX_PROMPT_CHARS` (default: `55000`, switches snapshot extraction to chunk-batched mode when prompt is too large)
- `HR_REPORT_SNAPSHOT_CHUNK_BATCH_SIZE` (default: `24`, chunk count per snapshot batch when batching is enabled)
- `HR_REPORT_URL_TIMEOUT_SECONDS` (default: `8`)
- `HR_REPORT_URL_CRAWL_LIMIT` (default: `25` child links per detected directory page)
- `HR_REPORT_URL_CRAWL_MAX_SEED_URLS` (default: `50`; disables directory crawling when you submit more seed URLs than this)
- `HR_REPORT_URL_MAX_TOTAL_FETCHES` (default: `120` total URL fetches per request)
- `HR_REPORT_URL_CHILD_WORKERS` (default: `6` for parallel child-page fetches)
- `HR_REPORT_PACK` (default: `tuning/packs/default_pack.yaml`)
- `HR_REPORT_INPUT_STORE` (default: `data/input_requests`)
- `HR_REPORT_OUTPUT_DIR` (default: `out`)
- `HR_REPORT_HOST` (default: `127.0.0.1`)
- `HR_REPORT_PORT` (default: `8080`)

## Run (Web, Recommended)
### Local
```bash
python -m app.main serve --host 127.0.0.1 --port 8080
```
Open `http://127.0.0.1:8080`.

### AWS/Lightsail
```bash
python -m app.main serve --host 0.0.0.0 --port 8080
```
Open `http://<your-public-ip>:8080`.

## UI Input Methods
The web UI accepts all three:
- pasted text
- URLs (paste/drop)
- uploaded files (drag-and-drop, browse, clipboard paste)

Pack/model selection is fixed by `.env` (`HR_REPORT_PACK`, `OPENAI_MODEL`) and not editable in the UI.

## API Endpoint
`POST /api/analyze` (multipart form)
- `text`
- `urls`
- `files[]`

Returns:
- `request_dir` (input persistence path)
- `result.output_dir`
- `result.report` + `result.report_markdown`

## Optional CLI Mode
```bash
python -m app.main run --pack tuning/packs/default_pack.yaml --input inputs/company_pack/
```

## Notes
- Disclaimer remains fixed: `This is not legal advice; consider reviewing with counsel.`
- If you changed packs previously under `packs/`, migrate to `tuning/packs/` for new defaults.
- Snapshot extraction runs as one LLM call first and only splits recursively on context-limit errors.
- Stage/scale inference uses explicit snapshot values first, then deterministic chunk signals (no separate headcount inference API step).
- Each run now writes `run_meta.json` in the output folder with ingestion/chunk/retrieval/LLM stats.

## Evidence Grounding Rules
- Findings use `evidence_status`:
  - `present`
  - `explicitly_missing`
  - `not_provided_in_sources`
  - `not_assessed`
- Retrieval classification is tracked per field:
  - `MENTIONED_EXPLICIT`
  - `MENTIONED_IMPLICIT`
  - `MENTIONED_AMBIGUOUS`
  - `NOT_FOUND_IN_RETRIEVED`
  - `NOT_RETRIEVED`
- `not_provided_in_sources` findings are confirmation-oriented and avoid “missing” language.
- Threshold prompts (headcount/complexity triggers) remain high-priority but use conditional wording.
- Top follow-up questions are deduplicated and capped.
- Report appendix lists reviewed sources for defensibility.
