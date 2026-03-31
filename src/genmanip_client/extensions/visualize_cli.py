from __future__ import annotations

import argparse


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "visualize",
        help="Browse and visualize eval results in the browser (with Rerun)",
        description=(
            "Starts a local web server and opens a browser UI for browsing runs, "
            "viewing per-task success rates, and visualizing episodes with the Rerun viewer."
        ),
    )
    p.add_argument(
        "--project_root",
        default=None,
        help="Project root containing saved/eval_results (default: cwd)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=55077,
        help="Local port for the web server (default: 55077)",
    )
    p.add_argument(
        "--flush-cache",
        action="store_true",
        help=(
            "Delete all cached .rrd visualizer files under the project root "
            "and exit. Use this to free disk space or force a rebuild."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="With --flush-cache: show what would be deleted without deleting.",
    )


def _flush_cache(project_root: str | None, dry_run: bool) -> int:
    from pathlib import Path

    root = Path(project_root).resolve() if project_root else Path.cwd()
    cache_name = ".genmanip_vis.rrd"

    files = sorted(root.rglob(cache_name))
    if not files:
        print(f"No cached .rrd files found under {root}")
        return 0

    action = "Would delete" if dry_run else "Deleting"
    total_bytes = 0
    for f in files:
        size = f.stat().st_size if f.exists() else 0
        total_bytes += size
        rel = f.relative_to(root)
        print(f"  {action}: {rel}  ({size / 1024 / 1024:.1f} MB)")

    print(f"\n{action} {len(files)} file(s), {total_bytes / 1024 / 1024:.1f} MB total.")
    if dry_run:
        return 0

    removed = 0
    for f in files:
        try:
            f.unlink()
            removed += 1
        except OSError as e:
            print(f"  Warning: could not delete {f}: {e}")

    print(f"Removed {removed} file(s).")
    return 0


def run(args: argparse.Namespace) -> int:
    if getattr(args, "flush_cache", False):
        return _flush_cache(args.project_root, getattr(args, "dry_run", False))
    from .visualize import run_visualizer
    run_visualizer(args.project_root, args.port)
    return 0
