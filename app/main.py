from __future__ import annotations

import argparse
import json
import os
import logging

from app.env_loader import load_env_file
from app.logging_setup import setup_logging
from app.pipeline import run_assessment_pipeline

logger = logging.getLogger(__name__)

def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _apply_local_defaults(args: argparse.Namespace) -> None:
    if not args.local_test and not args.offline_test:
        return

    local_input = os.getenv("HR_REPORT_LOCAL_INPUT", "inputs/company_pack/")
    if not args.input and not args.text and not args.url:
        args.input = local_input

    if args.offline_test:
        local_snapshot = os.getenv(
            "HR_REPORT_LOCAL_SNAPSHOT",
            "tests/golden/expected_snapshot_early.json",
        )
        if not args.snapshot_file:
            args.snapshot_file = local_snapshot
        args.no_llm_plan = True


def build_parser() -> argparse.ArgumentParser:
    pack_default = os.getenv("HR_REPORT_PACK", "tuning/packs/default_pack.yaml")
    input_default = os.getenv("HR_REPORT_INPUT")
    output_default = os.getenv("HR_REPORT_OUTPUT_DIR", "out")
    model_default = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    timeout_default = _env_int("OPENAI_TIMEOUT_SECONDS", 90)
    no_llm_plan_default = _env_bool("HR_REPORT_NO_LLM_PLAN", default=False)
    local_test_default = _env_bool("HR_REPORT_LOCAL_TEST", default=False)
    offline_test_default = _env_bool("HR_REPORT_OFFLINE_TEST", default=False)
    host_default = os.getenv("HR_REPORT_HOST", "127.0.0.1")
    port_default = _env_int("HR_REPORT_PORT", 8080)

    parser = argparse.ArgumentParser(description="HR Growth-Areas Assessment Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an assessment from CLI")
    run_parser.add_argument("--pack", default=pack_default, help="Path to consultant pack YAML/JSON")
    run_parser.add_argument("--input", default=input_default, help="File or directory with TXT/PDF/DOCX")
    run_parser.add_argument("--text", help="Pasted text input")
    run_parser.add_argument("--url", action="append", default=[], help="Optional URL(s) to ingest")
    run_parser.add_argument("--model", default=model_default, help="OpenAI model")
    run_parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=timeout_default,
        help="OpenAI request timeout in seconds",
    )
    run_parser.add_argument("--snapshot-file", help="Optional precomputed snapshot JSON")
    run_parser.add_argument(
        "--no-llm-plan",
        action="store_true",
        default=no_llm_plan_default,
        help="Use deterministic 30/60/90 plan generation",
    )
    run_parser.add_argument(
        "--local-test",
        action="store_true",
        default=local_test_default,
        help="Use local defaults while keeping production behavior",
    )
    run_parser.add_argument(
        "--offline-test",
        action="store_true",
        default=offline_test_default,
        help="Run without OpenAI calls using local snapshot",
    )
    run_parser.add_argument("--out", default=output_default, help="Output root directory")

    serve_parser = subparsers.add_parser("serve", help="Run web app for local/AWS usage")
    serve_parser.add_argument("--host", default=host_default, help="Bind host (e.g. 127.0.0.1 or 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=port_default, help="Bind port")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    return parser


def run_command(args: argparse.Namespace) -> int:
    _apply_local_defaults(args)
    logger.info(
        "cli_run_start pack=%s input=%s text=%s urls=%s out=%s",
        args.pack,
        args.input,
        bool(args.text),
        len(args.url or []),
        args.out,
    )
    result = run_assessment_pipeline(
        pack_path=args.pack,
        input_path=args.input,
        text=args.text,
        urls=args.url,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
        output_root=args.out,
        snapshot_file=args.snapshot_file,
        no_llm_plan=args.no_llm_plan,
    )
    print(json.dumps({"output_dir": result["output_dir"], "files": result["files"]}, indent=2))
    logger.info("cli_run_done output_dir=%s", result["output_dir"])
    return 0


def serve_command(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is required for web server mode") from exc

    uvicorn.run("app.web.server:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def main() -> int:
    load_env_file()
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        return run_command(args)
    if args.command == "serve":
        return serve_command(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
