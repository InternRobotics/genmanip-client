import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class RobotType(Enum):
    """Supported robot types."""

    FRANKA_PANDA_HAND = "manip/franka/panda_hand"
    FRANKA_ROBOTIQ = "manip/franka/robotiq"
    ALOHA_PIPER = "manip/mobile_aloha/piper"
    R5A_LIFT2 = "manip/lift2/R5a"


@dataclass
class RobotActionConfig:
    """Configuration for robot action slicing in visualization."""

    robot_type: RobotType
    is_dual_arm: bool
    # Action dimension info
    left_arm_slice: tuple[int, int]  # (start, end)
    left_gripper_slice: tuple[int, int]
    right_arm_slice: tuple[int, int] | None = None  # None for single arm
    right_gripper_slice: tuple[int, int] | None = None
    base_slice: tuple[int, int] | None = None  # None if no base
    total_dim: int = 0

    def __post_init__(self):
        if self.total_dim == 0:
            # Calculate total dimension
            end_vals = [self.left_arm_slice[1], self.left_gripper_slice[1]]
            if self.right_arm_slice:
                end_vals.append(self.right_arm_slice[1])
            if self.right_gripper_slice:
                end_vals.append(self.right_gripper_slice[1])
            if self.base_slice:
                end_vals.append(self.base_slice[1])
            self.total_dim = max(end_vals)


# Robot action configurations registry
ROBOT_ACTION_CONFIGS: dict[str, RobotActionConfig] = {
    # Franka with panda hand (single arm): 7 arm joints + 2 gripper = 9
    "manip/franka/panda_hand": RobotActionConfig(
        robot_type=RobotType.FRANKA_PANDA_HAND,
        is_dual_arm=False,
        left_arm_slice=(0, 7),
        left_gripper_slice=(7, 9),
        total_dim=9,
    ),
    # Franka with robotiq (single arm): 7 arm joints + 6 gripper = 13
    "manip/franka/robotiq": RobotActionConfig(
        robot_type=RobotType.FRANKA_ROBOTIQ,
        is_dual_arm=False,
        left_arm_slice=(0, 7),
        left_gripper_slice=(7, 13),
        total_dim=13,
    ),
    # Aloha with piper (dual arm): left(6+2) + right(6+2) + base(3) = 19
    "manip/mobile_aloha/piper": RobotActionConfig(
        robot_type=RobotType.ALOHA_PIPER,
        is_dual_arm=True,
        left_arm_slice=(0, 6),
        left_gripper_slice=(6, 8),
        right_arm_slice=(8, 14),
        right_gripper_slice=(14, 16),
        base_slice=(16, 19),
        total_dim=19,
    ),
    # R5a with lift2 (dual arm): left(6+2) + right(6+2) + base(3) = 19
    "manip/lift2/R5a": RobotActionConfig(
        robot_type=RobotType.R5A_LIFT2,
        is_dual_arm=True,
        left_arm_slice=(0, 6),
        left_gripper_slice=(6, 8),
        right_arm_slice=(8, 14),
        right_gripper_slice=(14, 16),
        base_slice=(16, 19),
        total_dim=19,
    ),
}


def get_robot_action_config(robot_id: str) -> RobotActionConfig | None:
    """Get action config for a robot type. Returns None if not found."""
    return ROBOT_ACTION_CONFIGS.get(robot_id)


def concat_cams_top(
    frames_by_cam: dict[str, np.ndarray],
    cam_order: list[str] | None = None,
) -> np.ndarray:
    """Concatenate camera frames horizontally (RGB)."""
    cams = cam_order[:] if cam_order else sorted(frames_by_cam.keys())
    cams = [c for c in cams if c in frames_by_cam]

    if not cams:
        raise RuntimeError("No camera frames provided for this step.")

    ref = frames_by_cam[cams[0]]
    h, w = ref.shape[:2]

    imgs = []
    for c in cams:
        im = frames_by_cam[c]
        if im.shape[1] != w or im.shape[0] != h:
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
        imgs.append(im)

    top = cv2.hconcat(imgs)  # RGB
    return top


class StreamingEpisodeRecorder:
    """
    Stream video writing per step:
      - Top: concat multiple cameras horizontally
      - Bottom: action plot (with vertical cursor at current time)
    """

    def __init__(
        self,
        out_dir: str,
        client_uid: str = "",
        fps: int = 30,
        plot_height: int = 480,
        video_scale=0.6,
        cam_order: list[str] | None = None,
        title: str = "Action",
        robot_id: str | None = None,
        frame_save_interval: int = 10,
        frame_dir_name: str = "images",
    ):
        self.out_dir = out_dir
        self.client_uid = client_uid
        self.fps = fps
        self.plot_height = plot_height
        self.video_scale = video_scale
        self.cam_order = cam_order or []
        self.title = title
        self.robot_id = robot_id
        self.frame_save_interval = frame_save_interval
        self.frame_dir_name = frame_dir_name
        self._robot_config: RobotActionConfig | None = (
            get_robot_action_config(robot_id) if robot_id else None
        )

        self._writer: cv2.VideoWriter | None = None
        self._episode_dir: str | None = None
        self._frame_dir: str | None = None
        self._t = 0
        self._actions: list[list[float]] = []
        self._states: list[list[float]] = []
        self._action_dim: int | None = None
        self._state_dim: int | None = None
        self._frame_w: int | None = None
        self._frame_h: int | None = None
        self._current_episode: str | None = None

    # ---------- helpers ----------
    def _process_state(self, state_dict: dict[str, Any]) -> list[float]:
        if not state_dict or self._robot_config is None:
            return []
        
        cfg = self._robot_config
        # Extract components based on expected keys
        joints = np.array(state_dict.get("state.joints", []))
        gripper = np.array(state_dict.get("state.gripper", []))
        base = np.array(state_dict.get("state.base", []))
        
        vec = []
        
        if cfg.is_dual_arm:
            # Assume equal split for dual arm
            n_j = len(joints) // 2
            n_g = len(gripper) // 2
            
            l_arm = joints[:n_j]
            r_arm = joints[n_j:]
            l_grip = gripper[:n_g]
            r_grip = gripper[n_g:]
            
            vec.extend(l_arm)
            vec.extend(l_grip)
            vec.extend(r_arm)
            vec.extend(r_grip)
        else:
            vec.extend(joints)
            vec.extend(gripper)
            
        if cfg.base_slice:
            vec.extend(base)
            
        return [float(x) for x in vec]

    def _flatten_action(self, action: Any) -> list[float]:
        out: list[float] = []
        if action is None:
            return out
        if isinstance(action, (int, float, np.number)):
            return [float(action)]
        if isinstance(action, (list, tuple)):
            for x in action:
                out.extend(self._flatten_action(x))
            return out
        if isinstance(action, np.ndarray):
            # flatten to 1D, then recurse on each element
            for x in action.reshape(-1):
                out.extend(self._flatten_action(x))
            return out
        if isinstance(action, dict):
            # stable order
            for k in sorted(action.keys()):
                out.extend(self._flatten_action(action[k]))
            return out
        return out

    def _ensure_writer(self, episode_id: str, top_h: int, top_w: int):
        # if new episode：close old writer
        if self._current_episode is not None and episode_id != self._current_episode:
            self.close()

        if self._writer is not None:
            return

        self._current_episode = episode_id
        self._episode_dir = os.path.join(self.out_dir, episode_id)
        Path(self._episode_dir).mkdir(parents=True, exist_ok=True)
        if self.frame_save_interval > 0:
            self._frame_dir = os.path.join(self._episode_dir, self.frame_dir_name)
            Path(self._frame_dir).mkdir(parents=True, exist_ok=True)

        final_h = top_h + self.plot_height
        final_w = top_w

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        suffix = f"_{self.client_uid}" if self.client_uid else ""
        out_path = os.path.join(self._episode_dir, f"merged_with_plot{suffix}.mp4")
        self._writer = cv2.VideoWriter(out_path, fourcc, self.fps, (final_w, final_h))

        self._frame_w = final_w
        self._frame_h = final_h

    def _resize_to(self, img: np.ndarray, w: int, h: int) -> np.ndarray:
        if img.shape[1] == w and img.shape[0] == h:
            return img
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    def _concat_cams_top(self, frames_by_cam: dict[str, np.ndarray]) -> np.ndarray:
        return concat_cams_top(frames_by_cam, self.cam_order)

    def _render_plot_rgb(self) -> np.ndarray:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_w = int(self._frame_w or 640)
        plot_h = int(self.plot_height)

        if not getattr(self, "_actions", None) and not getattr(self, "_states", None):
            return np.zeros((plot_h, plot_w, 3), dtype=np.uint8)

        # Helper to plot data
        def plot_data(ax_list, data_arr, t, title_prefix, cfg):
            if data_arr.ndim != 2:
                return

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

                if cfg.right_arm_slice and cfg.right_gripper_slice:
                    right_j = data_arr[:, cfg.right_arm_slice[0] : cfg.right_arm_slice[1]]
                    right_g = data_arr[:, cfg.right_gripper_slice[0] : cfg.right_gripper_slice[1]]
                else:
                    right_j = None
                    right_g = None

                has_base = cfg.base_slice is not None
                
                # 1) joints
                ax1 = ax_list[0]
                ax1.plot(left_j, label="left")
                if right_j is not None:
                    ax1.plot(right_j, label="right", linestyle="--")
                ax1.axvline(x=t, linewidth=2)
                ax1.set_title(f"{title_prefix} Joints")
                ax1.set_ylabel("joint")
                ax1.grid(True, alpha=0.2)

                # 2) grippers
                ax2 = ax_list[1]
                ax2.plot(left_g, label="left")
                if right_g is not None:
                    ax2.plot(right_g, label="right", linestyle="--")
                ax2.axvline(x=t, linewidth=2)
                ax2.set_title(f"{title_prefix} Grippers")
                ax2.set_ylabel("grip")
                ax2.grid(True, alpha=0.2)

                # 3) base
                if has_base and cfg.base_slice and len(ax_list) > 2:
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

            else:
                # Single arm
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

        # Prepare data
        actions_arr = np.asarray(self._actions, dtype=np.float32) if self._actions else None
        states_arr = np.asarray(self._states, dtype=np.float32) if self._states else None
        
        t = 0
        if actions_arr is not None and actions_arr.size > 0:
            t = actions_arr.shape[0] - 1
        elif states_arr is not None and states_arr.size > 0:
            t = states_arr.shape[0] - 1

        if self.robot_id is not None and self._robot_config is None:
            self._robot_config = get_robot_action_config(self.robot_id)
        cfg = self._robot_config

        dpi = 100
        fig_w_in = plot_w / dpi
        fig_h_in = plot_h / dpi

        # Determine layout
        has_states = states_arr is not None and states_arr.size > 0
        has_actions = actions_arr is not None and actions_arr.size > 0
        
        cols = 2 if (has_states and has_actions) else 1
        
        if cfg is None:
            rows = 1
        elif cfg.is_dual_arm and cfg.base_slice:
            rows = 3
        else:
            rows = 2
            
        fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
        gs = fig.add_gridspec(rows, cols)

        if has_states:
            col_idx = 0
            ax_list = [fig.add_subplot(gs[i, col_idx]) for i in range(rows)]
            plot_data(ax_list, states_arr, t, "State", cfg)
            
        if has_actions:
            col_idx = 1 if has_states else 0
            ax_list = [fig.add_subplot(gs[i, col_idx]) for i in range(rows)]
            plot_data(ax_list, actions_arr, t, "Pred Action", cfg)

        fig.tight_layout(pad=0.4)

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = rgba[:, :, :3].copy()

        plt.close(fig)

        if rgb.shape[0] != plot_h or rgb.shape[1] != plot_w:
            rgb = cv2.resize(rgb, (plot_w, plot_h), interpolation=cv2.INTER_AREA)

        return rgb

    # ---------- public API ----------
    def write_step(
        self,
        episode_id: str,
        frames_by_cam: dict[str, np.ndarray],
        action: Any,
        state: Any = None,
        robot_id: str | None = None,
    ):
        """
        frames_by_cam: {cam_name: RGB uint8 ndarray(H,W,3)}
        action: your action dict (nested ok)
        """
        if robot_id is not None:
            self.robot_id = robot_id
        top = self._concat_cams_top(frames_by_cam)
        if self.video_scale != 1.0:
            h, w = top.shape[:2]
            new_w = max(2, int(w * self.video_scale))
            new_h = max(2, int(h * self.video_scale))
            top = cv2.resize(top, (new_w, new_h), interpolation=cv2.INTER_AREA)
        top_h, top_w = top.shape[:2]

        # initialize writer once
        self._ensure_writer(episode_id, top_h, top_w)

        # record action (keep only numeric history, not images)
        flat = self._flatten_action(action)

        if flat:
            if self._action_dim is None:
                self._action_dim = len(flat)
            # If dim changes, truncate to min to avoid plot crash
            d = min(len(flat), self._action_dim)
            flat = flat[:d]
            self._actions.append(flat)
        else:
            # still advance timeline with zeros if you want; here we keep cursor consistent
            if self._action_dim is not None:
                self._actions.append([0.0] * self._action_dim)

        # record state
        if isinstance(state, dict):
            state_vec = self._process_state(state)
        else:
            state_vec = []
            
        if state_vec:
            if self._state_dim is None:
                self._state_dim = len(state_vec)
            d = min(len(state_vec), self._state_dim)
            state_vec = state_vec[:d]
            self._states.append(state_vec)
        else:
            if self._state_dim is not None:
                self._states.append([0.0] * self._state_dim)

        plot = self._render_plot_rgb()

        # Compose final frame: top + plot (both RGB)
        final_rgb = cv2.vconcat([top, plot])

        # write expects BGR
        final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)

        if self._writer is None:
            raise RuntimeError("VideoWriter not initialized (unexpected).")
        self._writer.write(final_bgr)

        if (
            self.frame_save_interval > 0
            and (self._t + 1) % self.frame_save_interval == 0
            and self._frame_dir is not None
        ):
            frame_path = os.path.join(self._frame_dir, f"frame_{self._t + 1:06d}.png")
            cv2.imwrite(frame_path, final_bgr)

        self._t += 1

    def close(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self._t = 0
        self._actions.clear()
        self._states.clear()
        self._action_dim = None
        self._state_dim = None
        self._frame_w = None
        self._frame_h = None
        self._episode_dir = None
        self._frame_dir = None
        self._current_episode = None
