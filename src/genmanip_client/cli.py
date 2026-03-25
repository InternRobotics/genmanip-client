from __future__ import annotations

import argparse
import importlib
import os
import sys

def _load_module(module_name: str):
    return importlib.import_module(module_name, package=__package__)

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
    _load_module(".submit_cli").register(subparsers)
    _load_module(".eval_cli").register(subparsers)
    _load_module(".plot_cli").register(subparsers)
    _load_module(".status_cli").register(subparsers)
    _load_module(".clean_cli").register(subparsers)

    # Extension subcommands
    _load_module(".extensions.online_cli").register(subparsers)
    if os.environ.get("GENMANIP_ENABLE_INTERNAL") == "1":
        _load_module(".extensions.leaderboard_cli").register(subparsers)

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
        return _load_module(".submit_cli").run(args)
    elif args.command == "eval":
        return _load_module(".eval_cli").run(args)
    elif args.command == "plot":
        return _load_module(".plot_cli").run(args)
    elif args.command == "status":
        return _load_module(".status_cli").run(args)
    elif args.command == "clean":
        return _load_module(".clean_cli").run(args)
    # Extension commands
    elif args.command == "online":
        return _load_module(".extensions.online_cli").run(args)
    elif args.command == "leaderboard":
        return _load_module(".extensions.leaderboard_cli").run(args)
    else:
        parser.print_help()
        return 1
