from __future__ import annotations

import argparse
import os
import sys

from .extensions import leaderboard_cli, online_cli
from . import eval_cli, plot_cli, submit_cli, status_cli

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

    # Core subcommands
    submit_cli.register(subparsers)
    eval_cli.register(subparsers)
    plot_cli.register(subparsers)
    status_cli.register(subparsers)

    # Extension subcommands
    online_cli.register(subparsers)
    if os.environ.get("GENMANIP_ENABLE_INTERNAL") == "1":
        leaderboard_cli.register(subparsers)

    # Parse arguments
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        if os.environ.get("GENMANIP_ENABLE_INTERNAL") != "1":
            print(
                "\nHint: set GENMANIP_ENABLE_INTERNAL=1 to enable internal commands.",
                file=sys.stderr,
            )
        return 0

    if args.command == "submit":
        return submit_cli.run(args)
    elif args.command == "eval":
        return eval_cli.run(args)
    elif args.command == "plot":
        return plot_cli.run(args)
    elif args.command == "status":
        return status_cli.run(args)
    # Extension commands
    elif args.command == "online":
        return online_cli.run(args)
    elif args.command == "leaderboard":
        return leaderboard_cli.run(args)
    else:
        parser.print_help()
        return 1
