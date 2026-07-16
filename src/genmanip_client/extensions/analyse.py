"""Data loading + aggregation for `gmp analyse`.

The bundled file ``default_cluster_map.json`` carries the EBench-v0.2-Generalist
task taxonomy (Operating Mode / Range / Precision / Atomic-Skill / Scene). It
is loaded automatically when no ``--cluster-map`` is supplied; clusters whose
tasks are absent from the actual data are pruned away so we only chart what we
can compute.


This module is intentionally framework-free: it only depends on the standard
library so the report can be produced on any machine that has the eval results
on disk (no plotting libs required at runtime — Plotly is loaded via CDN inside
the HTML template).

Local run layout (mirroring genmanip_client's eval_client output):

    <run_dir>/
        <task_name>/
            episode_result.json       # {seed: {"score": float, "sr": float}}
            <episode_id>/result.json  # {"score": float, "sr": float}
        ...

Task names typically embed a split suffix (`_val_train`, `_val_unseen`,
`_test_mini`); we strip it and keep the bare task name plus a `split` field.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---- bundled defaults ------------------------------------------------------

_BUNDLED_CLUSTER_MAP = Path(__file__).parent / "default_cluster_map.json"
_BUNDLED_PAYLOAD     = Path(__file__).parent / "default_payload.json"


def load_default_payload() -> dict[str, Any] | None:
    """Return the bundled EBench v3 reference payload (in gmp format), or None."""
    if not _BUNDLED_PAYLOAD.is_file():
        return None
    try:
        with _BUNDLED_PAYLOAD.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def merge_reference(payload: dict[str, Any], reference: dict[str, Any] | None = None
                    ) -> dict[str, Any]:
    """Merge the bundled EBench reference data into a locally-built payload so
    the report shows local runs + EBench models side-by-side. Local data wins
    on collisions; the merged result keeps the bundled cluster_map / cat_display
    / default_groups so the breakdown sections light up automatically.
    """
    if reference is None:
        reference = load_default_payload()
    if reference is None:
        return payload

    out = dict(payload)
    # runs: preserve order — local first, then any bundled runs not already present.
    seen = set(out.get("runs", []))
    out["runs"] = list(out.get("runs", []))
    for r in reference.get("runs", []):
        if r not in seen:
            out["runs"].append(r); seen.add(r)

    # tasks: union (preserve local order, append bundled extras)
    seen_t = set(out.get("tasks", []))
    out["tasks"] = list(out.get("tasks", []))
    for t in reference.get("tasks", []):
        if t not in seen_t:
            out["tasks"].append(t); seen_t.add(t)

    # splits: prefer local; pad with anything new from bundled
    seen_s = set(out.get("splits", []))
    out["splits"] = list(out.get("splits", []))
    for s in reference.get("splits", []):
        if s not in seen_s:
            out["splits"].append(s); seen_s.add(s)

    # Per-run aggregates: copy bundled rows that don't collide with local runs.
    for sect in ("agg_top", "agg_task", "agg_cluster"):
        out.setdefault(sect, {})
        for run, val in (reference.get(sect) or {}).items():
            if run not in out[sect]:
                out[sect][run] = val

    # cluster_map / cat_display: merge — bundled adds categories the local
    # data doesn't define; local takes precedence on overlapping keys.
    bundled_cm = reference.get("cluster_map") or {}
    if bundled_cm:
        merged_cm = dict(bundled_cm)
        merged_cm.update(out.get("cluster_map") or {})
        out["cluster_map"] = merged_cm
    bundled_disp = reference.get("cat_display") or {}
    if bundled_disp:
        merged_disp = dict(bundled_disp)
        merged_disp.update(out.get("cat_display") or {})
        out["cat_display"] = merged_disp

    # Carry the bundled default_groups so the EBench reference is visible the
    # moment the page loads, alongside the local checkboxes.
    if "default_groups" not in out and reference.get("default_groups"):
        out["default_groups"] = reference["default_groups"]
    return out


def load_default_cluster_map() -> dict[str, dict[str, Any]] | None:
    """Return the EBench taxonomy bundled next to this module, or None."""
    if not _BUNDLED_CLUSTER_MAP.is_file():
        return None
    try:
        with _BUNDLED_CLUSTER_MAP.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def prune_cluster_map(cluster_map: dict[str, dict[str, Any]],
                      present_tasks: set[str]) -> dict[str, dict[str, Any]]:
    """Drop clusters whose tasks aren't in `present_tasks`; drop empty
    categories. Keeps the report focused on what we can actually compute."""
    out: dict[str, dict[str, Any]] = {}
    for cat, clusters in cluster_map.items():
        kept: dict[str, Any] = {}

        if cat == "atomic_skill":
            for task, skill_tree in clusters.items():
                if task in present_tasks:
                    kept[task] = skill_tree
            if kept:
                out[cat] = kept
            continue

        for cn, task_list in clusters.items():
            inter = [t for t in task_list if t in present_tasks]
            if inter:
                kept[cn] = inter
        if kept:
            out[cat] = kept
    return out


def _collect_atomic_skill_observations(
    metric_scores: list[Any], skill_groups: list[Any]
) -> list[tuple[str, float, float]]:
    observations: list[tuple[str, float, float]] = []
    reached = True
    for score_stage, skill_stage in zip(metric_scores, skill_groups):
        if not reached:
            break
        if not isinstance(score_stage, list) or not isinstance(skill_stage, list):
            break
        stage_success = False
        alt_success: list[bool] = []
        for alt_scores in score_stage:
            success = (
                isinstance(alt_scores, list) and alt_scores and
                all(isinstance(v, (int, float)) and v != 0 for v in alt_scores)
            )
            alt_success.append(success)
            if success:
                stage_success = True

        if stage_success:
            for alt_success_flag, alt_skills in zip(alt_success, skill_stage):
                if not alt_success_flag or not isinstance(alt_skills, list):
                    continue
                for skill_list in alt_skills:
                    if not isinstance(skill_list, list):
                        continue
                    for skill in skill_list:
                        if isinstance(skill, str):
                            observations.append((skill, 1.0, 1.0))
        else:
            for alt_skills in skill_stage:
                if not isinstance(alt_skills, list):
                    continue
                for skill_list in alt_skills:
                    if not isinstance(skill_list, list):
                        continue
                    for skill in skill_list:
                        if isinstance(skill, str):
                            observations.append((skill, 0.0, 0.0))

        reached = stage_success
    return observations


def _normalize_atomic_skill_cluster_map(task_map: dict[str, Any],) -> dict[str, list[str]]:
    def extract_skills(obj: Any) -> set[str]:
        if isinstance(obj, str):
            return {obj}

        if isinstance(obj, list):
            skills = set()
            for item in obj:
                skills.update(extract_skills(item))
            return skills

        return set()

    skills = set()
    for stages in task_map.values():
        skills.update(extract_skills(stages))

    return {skill: [skill] for skill in skills}


# Display names for the bundled taxonomy categories
_CAT_DISPLAY = {
    "mobility":     "Operating Mode",
    "atomic_skill": "Atomic Skill",
    "range":        "Horizon",
    "precision":    "Precision",
    "scene":        "Scene",
}


# ---- split detection -------------------------------------------------------

_SPLIT_SUFFIXES = ("val_train", "val_unseen", "test_mini")


def parse_task_name(task_name: str) -> tuple[str, str]:
    """Return (base_task, split). If no split suffix is found, split='unknown'."""
    for s in _SPLIT_SUFFIXES:
        if task_name.endswith(f"_{s}"):
            return task_name[: -len(s) - 1], s
    return task_name, "unknown"


# ---- local result loaders --------------------------------------------------

_TASK_KEY_RE = re.compile(r"^\(\d+/\d+\)(?:.*/)?(?P<task>[^/]+)$")


def _records_from_result_json(run_id: str, path: Path) -> list[dict[str, Any]]:
    """Layout A: a single <run>/result.json with aggregated per-task SR/Score.

    Keys look like '(5/5)ebench/table_top_manip/foo_val_train' -> {sr, score}.
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, dict):
        return []
    out: list[dict[str, Any]] = []
    for raw_key, vals in data.items():
        if not isinstance(vals, dict):
            continue
        m = _TASK_KEY_RE.match(raw_key)
        last = m.group("task") if m else raw_key.split("/")[-1]
        base, split = parse_task_name(last)
        try:
            sr = float(vals.get("sr", vals.get("success_rate")))
            score = float(vals["score"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append({
            "run_id": run_id, "task": base, "split": split,
            "seed": "agg", "sr": sr, "score": score,
        })
    return out


def _records_from_result_info(run_id: str, run_dir: Path) -> list[dict[str, Any]]:
    """Layout B: per-episode result_info.json scattered under
    <run>/<bench>/<category>/<task_with_split>/<seed>/result_info.json.
    """
    out: list[dict[str, Any]] = []
    for ri in run_dir.rglob("result_info.json"):
        try:
            with ri.open("r", encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(d, dict):
            continue
        try:
            sr = float(d.get("sr", d.get("success_rate")))
            score = float(d["score"])
        except (KeyError, TypeError, ValueError):
            continue
        metric_scores = d.get("log_info", {}).get("metric_score")
        # path: .../<task_with_split>/<seed>/result_info.json
        seed_dir = ri.parent
        task_dir = seed_dir.parent
        base, split = parse_task_name(task_dir.name)
        out.append({
            "run_id": run_id, "task": base, "split": split,
            "seed": seed_dir.name, "sr": sr, "score": score,
            "metric_scores": metric_scores,
        })
    return out


def _records_from_episode_result(run_id: str, run_dir: Path) -> list[dict[str, Any]]:
    """Layout C (original): <run>/<task>/episode_result.json (genmanip's own
    format from save_episode_result)."""
    out: list[dict[str, Any]] = []
    for task_dir in sorted(run_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        ep = task_dir / "episode_result.json"
        if not ep.is_file():
            continue
        try:
            with ep.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        base, split = parse_task_name(task_dir.name)
        for seed, vals in data.items():
            try:
                sr = float(vals["sr"])
                score = float(vals["score"])
            except (KeyError, TypeError, ValueError):
                continue
            out.append({
                "run_id": run_id, "task": base, "split": split,
                "seed": str(seed), "sr": sr, "score": score,
            })
    return out


def load_run_episode_results(run_dir: Path) -> list[dict[str, Any]]:
    """Walk a single run directory, returning per-(task,seed) records.

    Tries multiple known layouts (in order of preference):
      A. <run>/.../<task>/<seed>/result_info.json      — per-episode raw output
      B. <run>/result.json                             — gmp eval aggregated output
      C. <run>/<task>/episode_result.json              — genmanip_client's own output
    """
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run dir not found: {run_dir}")
    run_id = run_dir.name

    # Layout A: prefer per-episode records because they retain metric_score,
    # which is required for atomic-skill aggregation.
    recs = _records_from_result_info(run_id, run_dir)
    if recs:
        return recs

    # Layout B: fall back to the aggregated result when raw episodes are absent.
    rj = run_dir / "result.json"
    if rj.is_file():
        recs = _records_from_result_json(run_id, rj)
        if recs:
            return recs

    # Layout C: episode_result.json under each task subdir
    return _records_from_episode_result(run_id, run_dir)


def discover_runs(project_root: str | os.PathLike | None,
                  benchmark_id: str | None = None) -> list[Path]:
    """Return all run directories under project_root/saved/eval_results/."""
    root = Path(project_root or os.getcwd()).resolve()
    base = root / "saved" / "eval_results"
    if not base.is_dir():
        return []
    runs: list[Path] = []
    for bench in sorted(base.iterdir()):
        if not bench.is_dir():
            continue
        if benchmark_id and bench.name != benchmark_id:
            continue
        for run in sorted(bench.iterdir()):
            if run.is_dir():
                runs.append(run)
    return runs


def resolve_run_paths(args_paths: list[str], project_root: str | None) -> list[Path]:
    """Convert CLI-style positional args into concrete run dirs.

    - If a positional arg is itself a directory containing at least one
      `*/episode_result.json`, treat it as a run dir.
    - Otherwise treat it as a run_id and search under project_root/saved/eval_results.
    - With no args, return all runs found under project_root.
    """
    if not args_paths:
        return discover_runs(project_root)
    resolved: list[Path] = []
    root = Path(project_root or os.getcwd()).resolve()
    for p in args_paths:
        cand = Path(p).expanduser()
        if cand.is_dir():
            resolved.append(cand.resolve())
            continue
        # treat as run_id; search under project_root/saved/eval_results/*/<id>
        base = root / "saved" / "eval_results"
        if base.is_dir():
            hits = [d for d in base.glob(f"*/{p}") if d.is_dir()]
            resolved.extend(d.resolve() for d in hits)
    # de-dup, preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for r in resolved:
        if r in seen:
            continue
        seen.add(r)
        unique.append(r)
    return unique


# ---- experiment grouping ---------------------------------------------------

def parse_group_spec(spec: str) -> tuple[str, list[str]]:
    """Parse 'Tag=pattern' or 'Tag=pattern1,pattern2' from --group flag.

    Patterns are glob-style (fnmatch). Multiple patterns are OR-combined.
    """
    if "=" not in spec:
        raise ValueError(f"--group expects 'Tag=pattern', got: {spec!r}")
    tag, patterns = spec.split("=", 1)
    pats = [p.strip() for p in patterns.split(",") if p.strip()]
    if not tag.strip() or not pats:
        raise ValueError(f"--group needs a non-empty tag and pattern: {spec!r}")
    return tag.strip(), pats


def load_groups_file(path: str | os.PathLike) -> dict[str, dict[str, Any]]:
    """Load a groups config JSON file. Two accepted shapes:

    A) Compact: {tag: "pattern"} or {tag: ["pat1", "pat2"]}
    B) Verbose: {tag: {"runs": ["pat1", ...], "color": "#hex"}}
    Returns the verbose form internally.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: dict[str, dict[str, Any]] = {}
    for tag, val in raw.items():
        if isinstance(val, str):
            out[tag] = {"runs": [val]}
        elif isinstance(val, list):
            out[tag] = {"runs": list(val)}
        elif isinstance(val, dict):
            patterns = val.get("runs") or val.get("patterns") or []
            if isinstance(patterns, str):
                patterns = [patterns]
            entry = {"runs": list(patterns)}
            if "color" in val:
                entry["color"] = val["color"]
            if "label" in val:
                entry["label"] = val["label"]
            out[tag] = entry
        else:
            raise ValueError(f"groups[{tag!r}] has unsupported type {type(val).__name__}")
    return out


def assign_group(run_id: str, groups: dict[str, dict[str, Any]]) -> str | None:
    """Return the first group tag whose patterns match `run_id`, else None."""
    for tag, entry in groups.items():
        for pat in entry.get("runs", []):
            if fnmatch.fnmatchcase(run_id, pat):
                return tag
    return None


def apply_groups(records: list[dict[str, Any]],
                 groups: dict[str, dict[str, Any]],
                 drop_ungrouped: bool = False
                 ) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Rewrite each record's run_id to the matching group tag.

    Within a group, the original run_id is appended to the seed value so
    multiple runs in one group become independent "seeds" for the std bands.

    Returns (rewritten_records, members) where members maps tag -> [run_ids].
    """
    if not groups:
        return records, {}
    members: dict[str, list[str]] = defaultdict(list)
    out: list[dict[str, Any]] = []
    for r in records:
        tag = assign_group(r["run_id"], groups)
        if tag is None:
            if drop_ungrouped:
                continue
            tag = r["run_id"]  # ungrouped runs keep their identity
        if r["run_id"] not in members[tag]:
            members[tag].append(r["run_id"])
        nr = dict(r)
        nr["seed"] = f"{r['run_id']}::{r['seed']}"
        nr["run_id"] = tag
        out.append(nr)
    return out, dict(members)


# ---- aggregation -----------------------------------------------------------

def _meanstd(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return {"mean": round(mean, 4), "std": round(std, 4), "n": len(vals)}


def aggregate(records: list[dict[str, Any]],
              cluster_map: dict[str, dict[str, list[str]]] | None = None
              ) -> dict[str, Any]:
    """Aggregate records produced by load_run_episode_results.

    Returns:
        {
          "runs":  ["run_id_1", ...],                # in input order
          "tasks": [...],                            # union of bare task names
          "splits": [...],                           # subset of {val_train, val_unseen, test_mini}
          "agg_top": {run -> {f"{split}_sr": {mean,std,n}, f"{split}_score": ...}}
          "agg_task": {run -> {split -> {task -> {sr_mean, sr_std, score_mean, score_std, n}}}}
          "agg_cluster": {run -> {split -> {category -> {cluster -> {sr_mean,...}}}}}
                       (only when cluster_map provided)
        }
    """
    runs: list[str] = []
    seen_runs: set[str] = set()
    for r in records:
        if r["run_id"] not in seen_runs:
            seen_runs.add(r["run_id"])
            runs.append(r["run_id"])

    tasks: list[str] = sorted({r["task"] for r in records})
    splits_present: list[str] = sorted(
        {r["split"] for r in records if r["split"] != "unknown"}
    )
    if not splits_present:
        splits_present = ["unknown"]

    # Top-line: per (run, split) aggregate over all (task, seed) values
    agg_top: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    by_run_split = defaultdict(lambda: defaultdict(lambda: {"sr": [], "score": []}))
    for r in records:
        by_run_split[r["run_id"]][r["split"]]["sr"].append(r["sr"])
        by_run_split[r["run_id"]][r["split"]]["score"].append(r["score"])
    for run, splits in by_run_split.items():
        for split, vals in splits.items():
            agg_top[run][f"{split}_sr"]    = _meanstd(vals["sr"])
            agg_top[run][f"{split}_score"] = _meanstd(vals["score"])
        # also compute "overall" across all splits
        all_sr    = [v for s in splits.values() for v in s["sr"]]
        all_score = [v for s in splits.values() for v in s["score"]]
        agg_top[run]["overall_sr"]    = _meanstd(all_sr)
        agg_top[run]["overall_score"] = _meanstd(all_score)

    # Per-task per-split per-run mean over seeds
    raw = defaultdict(lambda: {"sr": [], "score": []})
    for r in records:
        raw[(r["run_id"], r["split"], r["task"])]["sr"].append(r["sr"])
        raw[(r["run_id"], r["split"], r["task"])]["score"].append(r["score"])
    agg_task: dict[str, dict[str, dict[str, dict[str, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for (run, split, task), v in raw.items():
        agg_task[run][split][task] = {
            "sr_mean":    _meanstd(v["sr"])["mean"],
            "sr_std":     _meanstd(v["sr"])["std"],
            "score_mean": _meanstd(v["score"])["mean"],
            "score_std":  _meanstd(v["score"])["std"],
            "n":          len(v["sr"]),
        }

    out: dict[str, Any] = {
        "runs": runs,
        "tasks": tasks,
        "splits": splits_present,
        "agg_top": dict(agg_top),
        "agg_task": {k: dict(v) for k, v in agg_task.items()},
    }

    if cluster_map:
        # cluster_map: {category: {cluster_name: [task1, task2, ...]}}
        agg_cluster_raw = defaultdict(list)
        # group records by (run, split, task)
        per_rst = defaultdict(lambda: {"sr": [], "score": []})
        for r in records:
            per_rst[(r["run_id"], r["split"], r["task"])]["sr"].append(r["sr"])
            per_rst[(r["run_id"], r["split"], r["task"])]["score"].append(r["score"])

        # skill-level observations for atomic_skill new structure:
        atomic_skill_records: list[dict[str, Any]] = []
        atomic_skill_task_map = cluster_map.get("atomic_skill")
        if atomic_skill_task_map:
            cluster_map["atomic_skill"] = _normalize_atomic_skill_cluster_map(
                atomic_skill_task_map
            )
            for r in records:
                task = r["task"]
                metric_scores = r.get("metric_scores")

                if task not in atomic_skill_task_map or not isinstance(metric_scores, list):
                    continue

                for skill, sr, score in _collect_atomic_skill_observations(
                    metric_scores, atomic_skill_task_map[task]
                ):
                    nr = dict(r)
                    nr["skill"] = skill
                    nr["sr"] = sr
                    nr["score"] = score
                    atomic_skill_records.append(nr)

        # mean per task across seeds, then mean across tasks in cluster
        per_task_mean: dict[tuple[str, str, str], dict[str, float]] = {}
        for k, v in per_rst.items():
            per_task_mean[k] = {
                "sr": statistics.mean(v["sr"]) if v["sr"] else 0.0,
                "score": statistics.mean(v["score"]) if v["score"] else 0.0,
            }
        for (run, split, task), means in per_task_mean.items():
            for cat, clusters in cluster_map.items():
                if cat == "atomic_skill":
                    continue
                for cn, task_list in clusters.items():
                    if task in task_list:
                        agg_cluster_raw[(run, split, cat, cn)].append(means)

        if atomic_skill_task_map:
            for r in atomic_skill_records:
                for cn, skill_list in cluster_map.get("atomic_skill", {}).items():
                    if r["skill"] in skill_list:
                        agg_cluster_raw[(r["run_id"], r["split"], "atomic_skill", cn)].append({
                            "sr": r["sr"], "score": r["score"]
                        })

        agg_cluster: dict[str, Any] = {}
        for (run, split, cat, cn), vals in agg_cluster_raw.items():
            srs = [v["sr"] for v in vals]
            scs = [v["score"] for v in vals]
            d = (
                agg_cluster.setdefault(run, {})
                           .setdefault(split, {})
                           .setdefault(cat, {})
            )
            d[cn] = {
                "sr_mean":    round(statistics.mean(srs), 4),
                "sr_std":     round(statistics.stdev(srs), 4) if len(srs) > 1 else 0.0,
                "score_mean": round(statistics.mean(scs), 4),
                "score_std":  round(statistics.stdev(scs), 4) if len(scs) > 1 else 0.0,
                "n":          len(vals),
            }
        out["agg_cluster"] = agg_cluster
        out["cluster_map"] = cluster_map
        out["cat_display"] = {k: _CAT_DISPLAY.get(k, k.replace('_', ' ').title())
                              for k in cluster_map}
    return out


# ---- HTML rendering --------------------------------------------------------

# A clean, distinct color per run (cycles)
_DEFAULT_COLORS = [
    "#9381FF",  # lilac
    "#FF8FA3",  # coral pink
    "#75CFB8",  # mint teal
    "#FFD670",  # butter yellow
    "#9DC0F5",  # sky blue
    "#F2A5C9",  # rose
    "#F8C794",  # peach
    "#A6E3A1",  # mint
]


def render_html(payload: dict[str, Any], title: str = "GenManip Analysis",
                template_path: str | os.PathLike | None = None,
                default_groups: list[dict[str, Any]] | None = None) -> str:
    """Return a self-contained HTML string with the inlined payload.

    ``default_groups`` (optional) seeds the report's interactive Groups panel
    with predefined experiment groups. Each entry: {tag, members:[run_id,...],
    color (optional)}. Users can still modify or reset them in the browser.
    """
    template_path = template_path or (
        Path(__file__).parent / "analyse_template.html"
    )
    with open(template_path, "r", encoding="utf-8") as f:
        tmpl = f.read()
    payload = dict(payload)
    if default_groups:
        payload["default_groups"] = default_groups
    payload["title"] = title
    json_blob = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    out = tmpl.replace("__TITLE__", title)
    # The template has `const P = /*__PAYLOAD__*/null;` so the unreplaced page
    # is still valid JS. Replace BOTH the marker AND the trailing `null` so we
    # don't end up with `{...}null;` which is a syntax error.
    if "/*__PAYLOAD__*/null" in out:
        out = out.replace("/*__PAYLOAD__*/null", json_blob, 1)
    else:
        out = out.replace("/*__PAYLOAD__*/", json_blob, 1)
    return out


def write_report(records: list[dict[str, Any]],
                 output_path: str | os.PathLike,
                 title: str = "GenManip Analysis",
                 cluster_map: dict | None = None,
                 template_path: str | os.PathLike | None = None,
                 default_groups: list[dict[str, Any]] | None = None) -> Path:
    payload = aggregate(records, cluster_map=cluster_map)
    html = render_html(payload, title=title, template_path=template_path,
                       default_groups=default_groups)
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def write_report_from_payload(payload: dict[str, Any],
                              output_path: str | os.PathLike,
                              title: str = "GenManip Analysis",
                              template_path: str | os.PathLike | None = None,
                              default_groups: list[dict[str, Any]] | None = None
                              ) -> Path:
    """Render directly from a pre-built payload (e.g. the bundled EBench
    reference dataset). Skips the aggregate() step entirely."""
    if default_groups is not None:
        payload = dict(payload)
        payload["default_groups"] = default_groups
    html = render_html(payload, title=title, template_path=template_path,
                       default_groups=payload.get("default_groups"))
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
