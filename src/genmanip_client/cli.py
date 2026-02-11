from __future__ import annotations

import argparse
import json
import sys

from .eval_client import build_argparser as build_eval_argparser, run_cli as run_eval_cli
from .leaderboard import (
    DEFAULT_LEADERBOARD_HOST,
    DEFAULT_LEADERBOARD_PORT,
    DEFAULT_USER_TOKEN,
)
from .online_client import OnlineEvaluationClient
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
        "--token",
        type=str,
        default=None,
        help="API token for authenticated eval servers",
    )
    eval_parser.add_argument(
        "--robot_id",
        type=str,
        default=None,
        help="Robot ID for action visualization",
    )
    eval_parser.add_argument(
        "--web_view",
        action="store_true",
        help="Start a lightweight web viewer for the stream",
    )
    eval_parser.add_argument(
        "--web_view_port",
        type=int,
        default=55090,
        help="Web viewer port (default: 55090)",
    )
    eval_parser.add_argument(
        "--web_view_interval",
        type=int,
        default=10,
        help="Show one frame every N steps (default: 10)",
    )
    eval_parser.add_argument(
        "--web_view_scale",
        type=float,
        default=1.0,
        help="Scale factor for web viewer frames (default: 1.0)",
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

    # Online evaluation subcommands
    online_parser = subparsers.add_parser(
        "online",
        help="Online evaluation API client",
        description="Submit and query online evaluation tasks",
    )
    online_subparsers = online_parser.add_subparsers(
        dest="online_command", help="Online commands"
    )

    online_create = online_subparsers.add_parser(
        "create",
        help="Create an online evaluation task",
        description="Create an online evaluation task (optional task_id)",
    )
    online_create.add_argument(
        "--base_url",
        required=True,
        help="Online server base URL (e.g. https://example.com)",
    )
    online_create.add_argument(
        "--token",
        required=True,
        help="API token for online evaluation",
    )
    online_create.add_argument(
        "--task_id",
        default=None,
        help="Optional task_id to reuse for re-run",
    )
    online_create.add_argument(
        "--model_name",
        default=None,
        help="Model name (e.g. internVLA)",
    )
    online_create.add_argument(
        "--model_type",
        default=None,
        help="Model type (e.g. VLA)",
    )
    online_create.add_argument(
        "--benchmark_set",
        default=None,
        help="Benchmark set (e.g. EBench)",
    )
    online_create.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout seconds (default: 30)",
    )

    online_ready = online_subparsers.add_parser(
        "ready",
        help="Query online task readiness",
        description="Check if online evaluation task is ready",
    )
    online_ready.add_argument(
        "--base_url",
        required=True,
        help="Online server base URL (e.g. https://example.com)",
    )
    online_ready.add_argument(
        "--token",
        required=True,
        help="API token for online evaluation",
    )
    online_ready.add_argument(
        "--task_id",
        required=True,
        help="Task ID to query",
    )
    online_ready.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout seconds (default: 30)",
    )

    online_submit = online_subparsers.add_parser(
        "submit",
        help="Create task and wait until ready",
        description="Create an online evaluation task and poll until ready",
    )
    online_submit.add_argument(
        "--base_url",
        required=True,
        help="Online server base URL (e.g. https://example.com)",
    )
    online_submit.add_argument(
        "--token",
        required=True,
        help="API token for online evaluation",
    )
    online_submit.add_argument(
        "--task_id",
        default=None,
        help="Optional task_id to reuse for re-run",
    )
    online_submit.add_argument(
        "--model_name",
        default=None,
        help="Model name (e.g. internVLA)",
    )
    online_submit.add_argument(
        "--model_type",
        default=None,
        help="Model type (e.g. VLA)",
    )
    online_submit.add_argument(
        "--benchmark_set",
        default=None,
        help="Benchmark set (e.g. EBench)",
    )
    online_submit.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Overall wait timeout seconds (default: no timeout)",
    )
    online_submit.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Polling interval seconds (default: 5)",
    )
    online_submit.add_argument(
        "--print_endpoint",
        action="store_true",
        help="Print only the ready endpoint (for command substitution)",
    )

    # Leaderboard submission/listing
    leaderboard_parser = subparsers.add_parser(
        "leaderboard",
        help="Leaderboard submission utilities",
        description="Submit/list evaluation results for the leaderboard",
    )
    leaderboard_subparsers = leaderboard_parser.add_subparsers(
        dest="leaderboard_command", help="Leaderboard commands"
    )
    leaderboard_list = leaderboard_subparsers.add_parser(
        "list",
        help="List available local results",
        description="List evaluation results under saved/eval_results",
    )
    leaderboard_list.add_argument(
        "--project_root",
        default=None,
        help="Project root containing saved/eval_results (default: cwd)",
    )

    leaderboard_submit = leaderboard_subparsers.add_parser(
        "submit",
        help="Submit a result to the leaderboard",
        description="Zip and submit a local evaluation result to the leaderboard",
    )
    leaderboard_submit.add_argument("--run_id", required=True, help="Run ID to submit")
    leaderboard_submit.add_argument(
        "--benchmark_id",
        default=None,
        help="Benchmark ID (optional, will search if not provided)",
    )
    leaderboard_submit.add_argument(
        "-n", "--submission_name", required=True, help="Name of the submission"
    )
    leaderboard_submit.add_argument(
        "-l", "--leaderboard_name", required=True, help="Leaderboard name"
    )
    leaderboard_submit.add_argument(
        "--user_token", default=DEFAULT_USER_TOKEN, help="User token for leaderboard"
    )
    leaderboard_submit.add_argument(
        "--host", default=DEFAULT_LEADERBOARD_HOST, help="Leaderboard host"
    )
    leaderboard_submit.add_argument(
        "--port", type=int, default=DEFAULT_LEADERBOARD_PORT, help="Leaderboard port"
    )
    leaderboard_submit.add_argument(
        "--project_root",
        default=None,
        help="Project root containing saved/eval_results (default: cwd)",
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
    elif args.command == "online":
        return run_online(args)
    elif args.command == "leaderboard":
        return run_leaderboard(args)
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


def run_online(args: argparse.Namespace) -> int:
    if args.online_command is None:
        print("Error: online subcommand required", file=sys.stderr)
        return 1
    client = OnlineEvaluationClient(
        base_url=args.base_url,
        token=args.token,
        timeout=args.timeout,
    )
    try:
        if args.online_command == "create":
            resp = client.create_task(
                task_id=args.task_id,
                model_name=args.model_name,
                model_type=args.model_type,
                benchmark_set=args.benchmark_set,
            )
            print(json.dumps(resp, indent=2))
            return 0
        if args.online_command == "ready":
            resp = client.ready(task_id=args.task_id)
            print(json.dumps(resp, indent=2))
            return 0
        if args.online_command == "submit":
            create_resp = client.create_task(
                task_id=args.task_id,
                model_name=args.model_name,
                model_type=args.model_type,
                benchmark_set=args.benchmark_set,
            )
            create_data = create_resp.get("data", {}) if isinstance(create_resp, dict) else {}
            task_id = create_data.get("task_id") or args.task_id
            if not task_id:
                raise RuntimeError("Missing task_id in create response")
            ready_resp = client.wait_until_ready(
                task_id=task_id,
                interval=args.interval,
                timeout=args.timeout,
                pretty=not args.print_endpoint,
            )
            ready_data = ready_resp.get("data", {}) if isinstance(ready_resp, dict) else {}
            endpoint = ready_data.get("endpoint")
            if args.print_endpoint:
                if not endpoint:
                    raise RuntimeError("Ready response missing endpoint")
                print(endpoint)
            else:
                print(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "endpoint": endpoint,
                            "response": ready_resp,
                        },
                        indent=2,
                    )
                )
            return 0
        print("Error: unknown online subcommand", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def run_leaderboard(args: argparse.Namespace) -> int:
    if args.leaderboard_command is None:
        print("Error: leaderboard subcommand required", file=sys.stderr)
        return 1
    from .leaderboard import list_results, submit_results

    if args.leaderboard_command == "list":
        list_results(args.project_root)
        return 0
    if args.leaderboard_command == "submit":
        if args.user_token is None:
            print(
                "Please set USER_TOKEN environment variable or provide it by --user-token.",
                file=sys.stderr,
            )
            return 1
        submit_results(
            args.user_token,
            args.run_id,
            args.submission_name,
            args.leaderboard_name,
            args.host,
            args.port,
            args.project_root,
            benchmark_id=args.benchmark_id,
        )
        return 0
    print("Error: unknown leaderboard subcommand", file=sys.stderr)
    return 1
