from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CleanTarget:
    path: Path
    category: str
    reason: str


DEFAULT_DIR_TARGETS: tuple[tuple[str, str], ...] = (
    ("saved/assets/mesh_data", "Generated mesh cache derived from assets and USDs"),
    ("saved/eval_results", "Evaluation outputs and progress state"),
    ("logs", "Server and worker log files"),
)

OPTIONAL_DIR_TARGETS: tuple[tuple[str, str], ...] = (
    (
        "saved/assets/collected_packages",
        "Downloaded benchmark package cache that can be fetched again",
    ),
)

TEMP_FILE_PATTERNS: tuple[str, ...] = ("*.lock", "*_soft.lock", "*.tmp", "*.tmp-*")


def register(subparsers: argparse._SubParsersAction) -> None:
    clean_parser = subparsers.add_parser(
        "clean",
        help="Clean GenManip runtime caches and temporary files",
        description=(
            "Remove GenManip runtime caches and stale temporary files. By default "
            "this cleans saved/assets/mesh_data, saved/eval_results, logs, and "
            "recursive lock/tmp leftovers under the workspace."
        ),
    )
    clean_parser.add_argument(
        "--project_root",
        type=str,
        default=".",
        help="GenManip-Sim workspace root (default: current directory)",
    )
    clean_parser.add_argument(
        "--all",
        action="store_true",
        help="Also remove downloaded benchmark package cache under saved/assets/collected_packages",
    )
    clean_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without deleting anything",
    )


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _resolve_under(project_root: Path, relative_path: str) -> Path:
    candidate = (project_root / relative_path).absolute()
    if not _is_relative_to(candidate, project_root):
        raise ValueError(f"Path escapes project root: {relative_path}")
    return candidate


def _collect_dir_targets(project_root: Path, include_all: bool) -> list[CleanTarget]:
    targets: list[CleanTarget] = []
    for relative_path, reason in DEFAULT_DIR_TARGETS:
        path = _resolve_under(project_root, relative_path)
        if path.exists():
            targets.append(CleanTarget(path=path, category="dir", reason=reason))

    if include_all:
        for relative_path, reason in OPTIONAL_DIR_TARGETS:
            path = _resolve_under(project_root, relative_path)
            if path.exists():
                targets.append(CleanTarget(path=path, category="download-cache", reason=reason))
    return targets


def _collect_temp_file_targets(project_root: Path) -> list[CleanTarget]:
    targets: list[CleanTarget] = []
    seen: set[Path] = set()
    for root, _dirs, files in os.walk(project_root, followlinks=True):
        root_path = Path(root)
        for file_name in files:
            path = (root_path / file_name).absolute()
            if not any(path.match(pattern) for pattern in TEMP_FILE_PATTERNS):
                continue
            if path in seen:
                continue
            seen.add(path)
            targets.append(
                CleanTarget(
                    path=path,
                    category="temp-file",
                    reason="Runtime lock or temporary file",
                )
            )
    return targets


def collect_clean_targets(project_root: Path, include_all: bool) -> list[CleanTarget]:
    project_root = project_root.absolute()
    targets = _collect_dir_targets(project_root, include_all)
    dir_paths = {target.path for target in targets}

    for target in _collect_temp_file_targets(project_root):
        if any(_is_relative_to(target.path, dir_path) for dir_path in dir_paths):
            continue
        targets.append(target)

    targets.sort(key=lambda item: (str(item.path), item.category))
    return targets


def _format_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _remove_target(target: CleanTarget) -> None:
    if target.path.is_dir():
        shutil.rmtree(target.path)
        return
    target.path.unlink()


def run(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).absolute()
    targets = collect_clean_targets(project_root, include_all=args.all)

    if not targets:
        print(f"No cleanable cache files found under {project_root}")
        return 0

    action = "Would remove" if args.dry_run else "Removing"
    print(f"{action} {len(targets)} path(s) under {project_root}:")
    for target in targets:
        kind = "dir" if target.path.is_dir() else "file"
        print(f" - [{kind}] {_format_path(target.path, project_root)}: {target.reason}")

    if args.dry_run:
        return 0

    removed = 0
    for target in targets:
        if not target.path.exists():
            continue
        _remove_target(target)
        removed += 1

    print(f"Removed {removed} path(s).")
    print("Kept client_results, saved/tasks, and saved/demonstrations by default.")
    if not args.all:
        print("Use --all to also remove saved/assets/collected_packages.")
    return 0
