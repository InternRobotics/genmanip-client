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


def run(args: argparse.Namespace) -> int:
    from .submit import get_server_status, print_status

    base_url = f"http://{args.host}:{args.port}"
    try:
        status = get_server_status(base_url)
        print_status(status)
        return 0
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
