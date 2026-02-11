from __future__ import annotations

import argparse
import json
import sys

from .online_client import OnlineEvaluationClient


def register(subparsers: argparse._SubParsersAction) -> None:
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
        help="Optional model name for leaderboard display",
    )
    online_create.add_argument(
        "--model_type",
        default=None,
        help="Optional model type for leaderboard display",
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
        help="Optional model name for leaderboard display",
    )
    online_submit.add_argument(
        "--model_type",
        default=None,
        help="Optional model type for leaderboard display",
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


def run(args: argparse.Namespace) -> int:
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
            create_data = (
                create_resp.get("data", {}) if isinstance(create_resp, dict) else {}
            )
            task_id = create_data.get("task_id") or args.task_id
            if not task_id:
                raise RuntimeError("Missing task_id in create response")
            ready_resp = client.wait_until_ready(
                task_id=task_id,
                interval=args.interval,
                timeout=args.timeout,
                pretty=not args.print_endpoint,
            )
            ready_data = (
                ready_resp.get("data", {}) if isinstance(ready_resp, dict) else {}
            )
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
