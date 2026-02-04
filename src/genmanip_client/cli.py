from __future__ import annotations

import argparse
import sys

from .eval_client import build_argparser as build_eval_argparser, run_cli as run_eval_cli
from .submit import build_argparser as build_submit_argparser, run_submit


def main_legacy(argv: list[str] | None = None) -> int:
    """Legacy entry point for genmanip-client command (backward compatible)."""
    parser: argparse.ArgumentParser = build_eval_argparser()
    args = parser.parse_args(argv)
    return run_eval_cli(args)


def main(argv: list[str] | None = None) -> int:
    """
    Main entry point for gmp command with subcommands.

    Usage:
        gmp submit configs/tasks/xxx.yml [--run-id RUN_ID] [--overwrite]
        gmp eval --worker_ids 0,1 --host 127.0.0.1 --port 8087
    """
    if argv is None:
        argv = sys.argv[1:]

    # Create main parser
    parser = argparse.ArgumentParser(
        prog="gmp",
        description="GenManip evaluation client CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Submit subcommand
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
        "--run-id",
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

    # Eval subcommand (legacy behavior)
    eval_parser = subparsers.add_parser(
        "eval",
        help="Run evaluation client (legacy mode)",
        description="Run evaluation client with specified workers",
    )
    # Add eval arguments from the original parser
    eval_parser.add_argument(
        "--worker_ids",
        type=lambda s: s.split(","),
        default=["0"],
        help="List of worker IDs, i.e. --worker_ids 0,1,2",
    )
    eval_parser.add_argument("--host", type=str, default="0.0.0.0")
    eval_parser.add_argument("--port", type=int, default=8087)
    eval_parser.add_argument("--reset", action="store_true")
    eval_parser.add_argument("-a", "--arm_type", type=str, default="franka")
    eval_parser.add_argument("-g", "--gripper_type", type=str, default="panda_hand")
    eval_parser.add_argument("-c", "--control_type", type=str, default="joint_position")
    eval_parser.add_argument(
        "--robot_id",
        type=str,
        default=None,
        help="Robot ID for action visualization",
    )

    # Status subcommand
    status_parser = subparsers.add_parser(
        "status",
        help="Get current job status from the server",
        description="Query the evaluation server for current job status",
    )
    status_parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)",
    )
    status_parser.add_argument(
        "--port",
        type=int,
        default=8087,
        help="Server port (default: 8087)",
    )

    # Parse arguments
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "submit":
        return run_submit(args)
    elif args.command == "eval":
        return run_eval_cli(args)
    elif args.command == "status":
        return run_status(args)
    else:
        parser.print_help()
        return 1


def run_status(args: argparse.Namespace) -> int:
    """Run the status command."""
    from .submit import get_server_status, print_status

    base_url = f"http://{args.host}:{args.port}"
    try:
        status = get_server_status(base_url)
        print_status(status)
        return 0
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
