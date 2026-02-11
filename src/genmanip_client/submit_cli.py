from __future__ import annotations

import argparse

from .submit import run_submit


def register(subparsers: argparse._SubParsersAction) -> None:
    submit_parser = subparsers.add_parser(
        "submit",
        help="Submit evaluation jobs to the server",
        description="Submit evaluation jobs to GenManip eval server",
    )
    submit_parser.add_argument(
        "config_paths",
        nargs="+",
        help="Config file path(s) to evaluate",
    )
    submit_parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Run ID for this evaluation (auto-generated if not provided)",
    )
    submit_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force restart even if a job is in progress or completed",
    )
    submit_parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)",
    )
    submit_parser.add_argument(
        "--port",
        type=int,
        default=8087,
        help="Server port (default: 8087)",
    )


def run(args: argparse.Namespace) -> int:
    return run_submit(args)
