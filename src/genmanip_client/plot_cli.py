from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING, Any
import tempfile
import time

import numpy as np

if TYPE_CHECKING:
    from .vis_utils import RobotActionConfig
else:
    RobotActionConfig = Any


def register(subparsers: argparse._SubParsersAction) -> None:
    plot_parser = subparsers.add_parser(
        "plot",
        help="Generate action/state plots from an episode directory",
        description="Read steps.jsonl from an episode directory and write plot PNGs",
    )
    plot_parser.add_argument("episode_dir", type=str, help="Episode result directory")
    plot_parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Optional filename prefix for output plots",
    )
    plot_parser.add_argument(
        "--plot-height",
        type=int,
        default=600,
        help="Plot panel height for merged video output",
    )


def _load_steps(episode_dir: Path) -> tuple[list[dict], str | None]:
    steps_path = episode_dir / "steps.jsonl"
    if not steps_path.exists():
        raise FileNotFoundError(f"steps.jsonl not found: {steps_path}")

    rows: list[dict] = []
    robot_id: str | None = None
    with steps_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(row)
            if robot_id is None:
                candidate = row.get("robot_id")
                if isinstance(candidate, str) and candidate.strip():
                    robot_id = candidate
    if not rows:
        raise ValueError(f"No step records found in {steps_path}")
    return rows, robot_id


def _plot_data(ax_list, data_arr: np.ndarray, title_prefix: str, cfg: RobotActionConfig | None) -> None:
    t = data_arr.shape[0] - 1
    _plot_data_at_t(ax_list, data_arr, title_prefix, cfg, t)


def _plot_data_at_t(
    ax_list,
    data_arr: np.ndarray,
    title_prefix: str,
    cfg: RobotActionConfig | None,
    t: int,
) -> None:
    if cfg is None:
        ax = ax_list[0]
        ax.plot(data_arr)
        ax.axvline(x=t, linewidth=2)
        ax.set_title(f"{title_prefix} (dim={data_arr.shape[1]})")
        ax.set_xlabel("step")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.2)
        return

    if cfg.is_dual_arm:
        left_j = data_arr[:, cfg.left_arm_slice[0] : cfg.left_arm_slice[1]]
        left_g = data_arr[:, cfg.left_gripper_slice[0] : cfg.left_gripper_slice[1]]
        right_j = (
            data_arr[:, cfg.right_arm_slice[0] : cfg.right_arm_slice[1]]
            if cfg.right_arm_slice
            else None
        )
        right_g = (
            data_arr[:, cfg.right_gripper_slice[0] : cfg.right_gripper_slice[1]]
            if cfg.right_gripper_slice
            else None
        )

        ax1 = ax_list[0]
        ax1.plot(left_j, label="left")
        if right_j is not None:
            ax1.plot(right_j, label="right", linestyle="--")
        ax1.axvline(x=t, linewidth=2)
        ax1.set_title(f"{title_prefix} Joints")
        ax1.set_ylabel("joint")
        ax1.grid(True, alpha=0.2)

        ax2 = ax_list[1]
        ax2.plot(left_g, label="left")
        if right_g is not None:
            ax2.plot(right_g, label="right", linestyle="--")
        ax2.axvline(x=t, linewidth=2)
        ax2.set_title(f"{title_prefix} Grippers")
        ax2.set_ylabel("grip")
        ax2.grid(True, alpha=0.2)

        if cfg.base_slice and len(ax_list) > 2:
            ax3 = ax_list[2]
            base = data_arr[:, cfg.base_slice[0] : cfg.base_slice[1]]
            ax3.plot(base)
            ax3.axvline(x=t, linewidth=2)
            ax3.set_title(f"{title_prefix} Base")
            ax3.set_xlabel("step")
            ax3.set_ylabel("base")
            ax3.grid(True, alpha=0.2)
        else:
            ax2.set_xlabel("step")
        return

    arm_j = data_arr[:, cfg.left_arm_slice[0] : cfg.left_arm_slice[1]]
    gripper = data_arr[:, cfg.left_gripper_slice[0] : cfg.left_gripper_slice[1]]

    ax1 = ax_list[0]
    ax2 = ax_list[1]
    ax1.plot(arm_j)
    ax1.axvline(x=t, linewidth=2)
    ax1.set_title(f"{title_prefix} Arm Joints")
    ax1.set_ylabel("joint")
    ax1.grid(True, alpha=0.2)

    ax2.plot(gripper)
    ax2.axvline(x=t, linewidth=2)
    ax2.set_title(f"{title_prefix} Gripper")
    ax2.set_xlabel("step")
    ax2.set_ylabel("grip")
    ax2.grid(True, alpha=0.2)


def _write_plot(out_path: Path, data: np.ndarray, title: str, cfg: RobotActionConfig | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if cfg is None:
        rows = 1
    elif cfg.is_dual_arm and cfg.base_slice:
        rows = 3
    else:
        rows = 2

    fig, axes = plt.subplots(rows, 1, figsize=(14, 3.5 * rows), squeeze=False)
    ax_list = [axes[i, 0] for i in range(rows)]
    _plot_data(ax_list, data, title, cfg)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _render_plot_panel_rgb(
    *,
    action_arr: np.ndarray | None,
    state_arr: np.ndarray | None,
    cfg: RobotActionConfig | None,
    t: int,
    width: int,
    height: int,
) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_states = state_arr is not None and state_arr.size > 0
    has_actions = action_arr is not None and action_arr.size > 0
    if not has_states and not has_actions:
        return np.zeros((height, width, 3), dtype=np.uint8)

    cols = 2 if (has_states and has_actions) else 1
    if cfg is None:
        rows = 1
    elif cfg.is_dual_arm and cfg.base_slice:
        rows = 3
    else:
        rows = 2

    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    gs = fig.add_gridspec(rows, cols)

    if has_states and state_arr is not None:
        axes = [fig.add_subplot(gs[i, 0]) for i in range(rows)]
        _plot_data_at_t(axes, state_arr, "State", cfg, min(t, state_arr.shape[0] - 1))

    if has_actions and action_arr is not None:
        col_idx = 1 if has_states else 0
        axes = [fig.add_subplot(gs[i, col_idx]) for i in range(rows)]
        _plot_data_at_t(axes, action_arr, "Action", cfg, min(t, action_arr.shape[0] - 1))

    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = rgba[:, :, :3].copy()
    plt.close(fig)
    return rgb


def _find_source_video(episode_dir: Path) -> Path | None:
    candidates = sorted(
        p
        for p in episode_dir.glob("merged*.mp4")
        if "_with_plot" not in p.stem
    )
    return candidates[0] if candidates else None


def _build_output_video_path(source_video: Path, prefix: str) -> Path:
    stem = source_video.stem
    suffix = stem[len("merged") :]
    return source_video.with_name(f"{prefix}merged_with_plot{suffix}.mp4")


def _ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def _run_ffmpeg(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )


def _transcode_to_compatible_mp4(
    source_video: Path,
    output_video: Path,
) -> None:
    ffmpeg = _ffmpeg_path()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found in PATH")

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(source_video),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(output_video),
    ]
    result = _run_ffmpeg(cmd)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            f"ffmpeg transcode failed for {source_video}: {detail or 'unknown error'}"
        )


def _write_plot_video(
    *,
    episode_dir: Path,
    source_video: Path,
    out_video: Path,
    action_arr: np.ndarray | None,
    state_arr: np.ndarray | None,
    cfg: RobotActionConfig | None,
    plot_height: int,
) -> None:
    import cv2

    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source video: {source_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_h = height + int(plot_height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    temp_output = out_video
    temp_dir_obj: tempfile.TemporaryDirectory[str] | None = None
    if _ffmpeg_path() is not None:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="gmp_plot_")
        temp_output = Path(temp_dir_obj.name) / out_video.name

    writer = cv2.VideoWriter(str(temp_output), fourcc, fps, (width, out_h))
    if not writer.isOpened():
        cap.release()
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()
        raise RuntimeError(f"Failed to open output video writer: {temp_output}")

    frame_idx = 0

    def print_progress(current: int) -> None:
        if total_frames > 0:
            width_chars = 30
            filled = min(width_chars, int(width_chars * current / total_frames))
            bar = "#" * filled + "-" * (width_chars - filled)
            msg = f"\rProcessing video [{bar}] {current}/{total_frames}"
        else:
            msg = f"\rProcessing video frames: {current}"
        print(msg, end="", file=sys.stderr, flush=True)

    try:
        while True:
            ok, top_bgr = cap.read()
            if not ok or top_bgr is None:
                break
            panel_rgb = _render_plot_panel_rgb(
                action_arr=action_arr,
                state_arr=state_arr,
                cfg=cfg,
                t=frame_idx,
                width=width,
                height=int(plot_height),
            )
            panel_bgr = cv2.cvtColor(panel_rgb, cv2.COLOR_RGB2BGR)
            final_bgr = cv2.vconcat([top_bgr, panel_bgr])
            writer.write(final_bgr)
            frame_idx += 1
            if frame_idx == 1 or frame_idx % 10 == 0 or (
                total_frames > 0 and frame_idx == total_frames
            ):
                print_progress(frame_idx)
    finally:
        cap.release()
        writer.release()
        if frame_idx > 0:
            print_progress(frame_idx)
            print(file=sys.stderr, flush=True)

    try:
        if temp_output != out_video:
            _transcode_to_compatible_mp4(temp_output, out_video)
            print(f"Wrote compatible plot video {out_video}")
        else:
            print(f"Wrote {out_video}")
    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()


def run(args: argparse.Namespace) -> int:
    from .vis_utils import ROBOT_ACTION_CONFIGS

    start_time = time.time()
    episode_dir = Path(args.episode_dir).expanduser().resolve()
    print(
        f"[gmp plot] start episode_dir={episode_dir} "
        f"at {time.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    rows, robot_id = _load_steps(episode_dir)
    cfg = ROBOT_ACTION_CONFIGS.get(robot_id) if robot_id else None

    actions = [row.get("action", []) for row in rows if row.get("action")]
    states = [row.get("state", []) for row in rows if row.get("state")]
    action_arr = np.asarray(actions, dtype=np.float32) if actions else None
    state_arr = np.asarray(states, dtype=np.float32) if states else None

    prefix = args.prefix
    if action_arr is not None:
        action_path = episode_dir / f"{prefix}action_plot.png"
        _write_plot(action_path, action_arr, "Action", cfg)
        print(f"Wrote {action_path}")
    else:
        print(f"No action data in {episode_dir / 'steps.jsonl'}")

    if state_arr is not None:
        state_path = episode_dir / f"{prefix}state_plot.png"
        _write_plot(state_path, state_arr, "State", cfg)
        print(f"Wrote {state_path}")
    else:
        print(f"No state data in {episode_dir / 'steps.jsonl'}")

    source_video = _find_source_video(episode_dir)
    if source_video is None:
        print(f"No merged source video found in {episode_dir}")
        return 0

    out_video = _build_output_video_path(source_video, prefix)
    _write_plot_video(
        episode_dir=episode_dir,
        source_video=source_video,
        out_video=out_video,
        action_arr=action_arr,
        state_arr=state_arr,
        cfg=cfg,
        plot_height=int(args.plot_height),
    )
    elapsed = time.time() - start_time
    print(
        f"[gmp plot] done episode_dir={episode_dir} "
        f"elapsed={elapsed:.2f}s"
    )
    return 0
