"""CLI registration for `gmp analyse`."""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "analyse",
        aliases=["analyze"],
        help="Build an HTML report from one or more eval runs",
        description=(
            "Aggregate local eval results (saved/eval_results) and write a "
            "self-contained HTML report with charts (top-line bars, radar, "
            "scatter, per-task heatmap, optional cluster breakdowns)."
        ),
    )
    p.add_argument(
        "runs",
        nargs="*",
        help=(
            "Run directories or run_ids to include. If omitted, every run under "
            "<project_root>/saved/eval_results is used."
        ),
    )
    p.add_argument(
        "--project_root",
        default=None,
        help="Project root containing saved/eval_results (default: cwd)",
    )
    p.add_argument(
        "-o", "--output",
        default="analyse_report.html",
        help="Output HTML file path (default: ./analyse_report.html)",
    )
    p.add_argument(
        "--title",
        default="GenManip Analysis",
        help="Title shown at the top of the report",
    )
    p.add_argument(
        "--cluster-map",
        default=None,
        help=(
            "Path to a JSON file mapping {category: {cluster_name: [task,...]}}."
            " If omitted, the EBench-v0.2-Generalist taxonomy bundled with the tool"
            " is auto-loaded; clusters whose tasks are absent from the data are"
            " pruned. Pass --no-cluster-map to skip cluster charts entirely."
        ),
    )
    p.add_argument(
        "--no-cluster-map",
        dest="no_cluster_map",
        action="store_true",
        help="Don't load any cluster map (skip cluster breakdown section).",
    )
    p.add_argument(
        "--reference",
        dest="reference",
        action="store_true",
        help=(
            "Force-render only the bundled EBench v0.2 reference payload "
            "(Pi0-200k / Pi0.5-200k / XVLA-200k / InternVLA-A1-200k); skip "
            "local saved/eval_results entirely."
        ),
    )
    p.add_argument(
        "--no-reference",
        dest="no_reference",
        action="store_true",
        help=(
            "Skip the bundled EBench reference and only show local runs. "
            "By default the report includes BOTH the local runs and the "
            "EBench reference models so they can be compared side-by-side."
        ),
    )
    p.add_argument(
        "-g", "--group",
        dest="group",
        action="append",
        default=[],
        metavar="TAG=GLOB",
        help=(
            "Define an experiment tag that bundles runs whose run_id matches "
            "the given glob pattern. Repeatable. Example: "
            "--group 'Pi0.5 sweep=pi05-tabletop-*' "
            "--group 'Pi0 baseline=pi0_chunkrel_*'. "
            "Use comma to OR multiple patterns: --group 'Tag=foo-*,bar-*'."
        ),
    )
    p.add_argument(
        "--groups",
        dest="groups_file",
        default=None,
        help=(
            "Path to a JSON file defining experiment tags. Either "
            "{'Tag': 'pat'} / {'Tag': ['pat1','pat2']} (compact), or "
            "{'Tag': {'runs':[...], 'color':'#hex', 'label':'...'}} (verbose)."
        ),
    )
    p.add_argument(
        "--open",
        action="store_true",
        help="Open the rendered report in the default browser when done.",
    )


def _emit_reference(args, reason: str) -> int:
    """Render the bundled EBench v0.2 reference payload directly."""
    from . import analyse
    payload = analyse.load_default_payload()
    if not payload:
        print("No bundled reference payload available.", file=sys.stderr)
        return 1
    print(f"  {reason} — rendering bundled EBench v0.2 reference data "
          f"({len(payload.get('runs',[]))} models, "
          f"{len(payload.get('tasks',[]))} tasks)")
    title = args.title
    if title == "GenManip Analysis":
        title = "GenManip · EBench v0.2 reference"
    out = analyse.write_report_from_payload(
        payload, output_path=args.output, title=title,
    )
    print(f"Report written: {out}")
    if args.open:
        webbrowser.open(out.as_uri())
    return 0


def run(args: argparse.Namespace) -> int:
    from . import analyse  # local import (keeps gmp startup fast)

    # --reference: skip local entirely and render the bundled payload
    if args.reference:
        return _emit_reference(args, "--reference")

    run_paths = analyse.resolve_run_paths(args.runs, args.project_root)
    if not run_paths:
        return _emit_reference(args, "no runs found in project root")

    records: list[dict] = []
    skipped_empty: list[Path] = []
    for rp in run_paths:
        try:
            recs = analyse.load_run_episode_results(rp)
        except FileNotFoundError as exc:
            print(f"warning: {exc}", file=sys.stderr)
            continue
        if not recs:
            skipped_empty.append(rp)
            continue
        print(f"  loaded {len(recs):4d} records  from {rp}")
        records.extend(recs)
    if skipped_empty:
        print(
            f"  (skipped {len(skipped_empty)} empty / in-progress run(s) — "
            f"no result.json or result_info.json found)",
            file=sys.stderr,
        )
        if len(skipped_empty) <= 12:
            for p in skipped_empty:
                print(f"    · {p}", file=sys.stderr)

    if not records:
        return _emit_reference(args, "no loadable local runs")

    # ---- experiment grouping (CLI flags become initial UI groups) ----
    groups_def: dict[str, dict] = {}
    if args.groups_file:
        groups_def.update(analyse.load_groups_file(args.groups_file))
    for spec in (args.group or []):
        try:
            tag, pats = analyse.parse_group_spec(spec)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        groups_def[tag] = {"runs": pats}
    # Resolve patterns -> concrete member run_ids using the actually loaded data.
    default_groups: list[dict] = []
    if groups_def:
        all_runs = sorted({r["run_id"] for r in records})
        for tag, entry in groups_def.items():
            members = []
            for r in all_runs:
                for pat in entry.get("runs", []):
                    import fnmatch as _fn
                    if _fn.fnmatchcase(r, pat):
                        members.append(r); break
            if members:
                g = {"tag": tag, "members": members}
                if "color" in entry:
                    g["color"] = entry["color"]
                default_groups.append(g)
                print(f"  group [{tag}]  pre-populated with {len(members)} run(s)")
    if default_groups:
        print("  (open the report and use the Experiment Groups panel to modify them)")

    cluster_map = None
    if args.cluster_map:
        with open(args.cluster_map, "r", encoding="utf-8") as f:
            cluster_map = json.load(f)
    elif not getattr(args, "no_cluster_map", False):
        cluster_map = analyse.load_default_cluster_map()
        if cluster_map is not None:
            present = {r["task"] for r in records}
            pruned = analyse.prune_cluster_map(cluster_map, present)
            if pruned:
                cluster_map = pruned
                cats = ", ".join(pruned.keys())
                print(f"  using bundled cluster map [{cats}]")
            else:
                cluster_map = None
                print("  bundled cluster map has no overlap with loaded tasks; skipping cluster charts")

    # Build the local payload first.
    payload = analyse.aggregate(records, cluster_map=cluster_map)

    # Merge the EBench reference unless explicitly opted out — this is what
    # makes the report show your submissions AND the bundled reference models
    # side-by-side, so you can compare directly.
    if not getattr(args, "no_reference", False):
        ref = analyse.load_default_payload()
        if ref is not None:
            payload = analyse.merge_reference(payload, ref)
            ref_runs = ref.get("runs", [])
            print(f"  + merged EBench reference ({len(ref_runs)} models: "
                  f"{', '.join(ref_runs)})")

    if default_groups:
        payload = dict(payload)
        # CLI-defined groups REPLACE the bundled defaults
        payload["default_groups"] = default_groups

    out = analyse.write_report_from_payload(
        payload,
        output_path=args.output,
        title=args.title,
    )
    print(f"Report written: {out}")
    if args.open:
        webbrowser.open(out.as_uri())
    return 0
