from __future__ import annotations

import argparse

from .eval_client import run_cli as run_eval_cli


def register(subparsers: argparse._SubParsersAction) -> None:
    eval_parser = subparsers.add_parser(
        "eval",
        help="Run evaluation client (legacy mode)",
        description="Run evaluation client with specified workers",
    )
    eval_parser.add_argument(
        "--worker_ids",
        type=lambda s: s.split(","),
        default=["0"],
        help="List of worker IDs, i.e. --worker_ids 0,1,2",
    )
    eval_parser.add_argument("--host", type=str, default="0.0.0.0")
    eval_parser.add_argument("--port", type=int, default=8087)
    eval_parser.add_argument("--url", type=str, default=None)
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
    eval_parser.add_argument(
        "--frame_save_interval",
        type=int,
        default=0,
        help="Save one image every N steps (0 disables, default: 0)",
    )


def run(args: argparse.Namespace) -> int:
    return run_eval_cli(args)
