from __future__ import annotations

import argparse
import sys

from .leaderboard import (
    DEFAULT_LEADERBOARD_HOST,
    DEFAULT_LEADERBOARD_PORT,
    DEFAULT_USER_TOKEN,
    list_results,
    submit_results,
)


def register(subparsers: argparse._SubParsersAction) -> None:
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
    leaderboard_submit.add_argument(
        "--include-videos",
        action="store_true",
        default=False,
        help="Include video files in the submission (larger upload)",
    )


def run(args: argparse.Namespace) -> int:
    if args.leaderboard_command is None:
        print("Error: leaderboard subcommand required", file=sys.stderr)
        return 1
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
            include_videos=args.include_videos,
        )
        return 0
    print("Error: unknown leaderboard subcommand", file=sys.stderr)
    return 1
