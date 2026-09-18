from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from pydantic import ValidationError

from .analysis import analyze
from .models import AnalysisConfig
from .providers import EvaluationError
from .reporting import write_report
from .sources import SourceError, load_source, validate_dataset


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip() or default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lecturer-feedback",
        description="Generate aggregate learning feedback from task-based chat logs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate and summarize an input export.")
    validate.add_argument(
        "--input", required=True, type=Path, help="SQLite, JSONL, or beta-log .json.gz file."
    )

    run = subparsers.add_parser("analyze", help="Analyze an export and write aggregate reports.")
    run.add_argument(
        "--input", required=True, type=Path, help="SQLite, JSONL, or beta-log .json.gz file."
    )
    run.add_argument("--output", required=True, type=Path, help="Directory for generated artifacts.")
    run.add_argument(
        "--provider",
        choices=("openai", "openrouter"),
        default=_env("LECTURER_FEEDBACK_PROVIDER", "openai"),
    )
    run.add_argument("--model", default=os.getenv("LECTURER_FEEDBACK_MODEL") or None)
    run.add_argument(
        "--language",
        choices=("de", "en"),
        default=_env("LECTURER_FEEDBACK_LANGUAGE", "de"),
    )
    run.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout in seconds.")
    run.add_argument("--retries", type=int, default=2, help="Retries after failed model responses.")
    run.add_argument(
        "--max-transcript-chars",
        type=int,
        default=50000,
        help="Maximum transcript characters sent per conversation.",
    )
    run.add_argument("--max-conversations", type=int, default=None, help="Optional pilot-run cap.")
    run.add_argument("--no-html", action="store_true", help="Do not generate the local HTML report.")
    return parser


def _validate(input_path: Path) -> int:
    dataset = load_source(input_path)
    candidates = sum(
        1
        for conversation in dataset.conversations
        if any(message.role == "user" and message.content.strip() for message in conversation.messages)
    )
    print(f"Source: {dataset.source_name}")
    print(f"Tasks: {len(dataset.tasks)}")
    print(f"Conversations: {len(dataset.conversations)}")
    print(f"Candidate conversations: {candidates}")
    for warning in validate_dataset(dataset):
        print(f"Warning: {warning}")
    return 0


def _analyze(args: argparse.Namespace) -> int:
    config = AnalysisConfig(
        provider=args.provider,
        model=args.model,
        language=args.language,
        timeout_seconds=args.timeout,
        max_retries=args.retries,
        max_transcript_chars=args.max_transcript_chars,
        max_conversations=args.max_conversations,
    )
    report = analyze(
        args.input,
        config,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    paths = write_report(report, args.output, include_html=not args.no_html)
    for path in paths:
        print(path.resolve())
    if not report.run.complete:
        print("Analysis completed with errors; inspect report.json.", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            return _validate(args.input)
        return _analyze(args)
    except (SourceError, EvaluationError, ValidationError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
