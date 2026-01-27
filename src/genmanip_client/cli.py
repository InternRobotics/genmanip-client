from __future__ import annotations

import argparse

from .eval_client import build_argparser, run_cli


def main(argv: list[str] | None = None) -> int:
    parser: argparse.ArgumentParser = build_argparser()
    args = parser.parse_args(argv)
    return run_cli(args)
