from __future__ import annotations

import argparse
import sys


def register(subparsers: argparse._SubParsersAction) -> None:
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
    status_parser.add_argument("--url", type=str, default=None)
    status_parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="API token for authenticated eval servers",
    )
    status_parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Optional run ID to include in request headers",
    )


def run(args: argparse.Namespace) -> int:
    from .submit import get_server_status, print_status

    if getattr(args, "url", None):
        base_url = args.url
    else:
        base_url = f"http://{args.host}:{args.port}"

    headers: dict[str, str] = {}
    if getattr(args, "token", None):
        headers["Authorization"] = f"Bearer {args.token}"
    if getattr(args, "run_id", None):
        headers["run_id"] = args.run_id

    params: dict[str, str] = {}
    if getattr(args, "run_id", None):
        params["run_id"] = args.run_id

    try:
        status = get_server_status(base_url, headers=headers, params=params)
        print_status(status)
        return 0
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
