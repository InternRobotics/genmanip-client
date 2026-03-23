import json
import os
import time
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
    Stream video writing per step with lightweight metadata persistence.
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
        self._meta_fh: Any | None = None
        self._t = 0
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

        meta_path = os.path.join(self._episode_dir, "steps.jsonl")
        self._meta_fh = open(meta_path, "a", encoding="utf-8")

        final_h = top_h
        final_w = top_w

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        suffix = f"_{self.client_uid}" if self.client_uid else ""
        out_path = os.path.join(self._episode_dir, f"merged{suffix}.mp4")
        self._writer = cv2.VideoWriter(out_path, fourcc, self.fps, (final_w, final_h))

        self._frame_w = final_w
        self._frame_h = final_h

    def _resize_to(self, img: np.ndarray, w: int, h: int) -> np.ndarray:
        if img.shape[1] == w and img.shape[0] == h:
            return img
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    def _concat_cams_top(self, frames_by_cam: dict[str, np.ndarray]) -> np.ndarray:
        return concat_cams_top(frames_by_cam, self.cam_order)

    def _append_step_metadata(
        self,
        *,
        episode_id: str,
        action: Any,
        state: Any,
        frames_by_cam: dict[str, np.ndarray],
    ) -> None:
        if self._meta_fh is None:
            return
        state_vec = self._process_state(state) if isinstance(state, dict) else []
        record = {
            "episode_id": episode_id,
            "step": self._t,
            "robot_id": self.robot_id,
            "camera_names": sorted(frames_by_cam.keys()),
            "action": self._flatten_action(action),
            "state": state_vec,
        }
        self._meta_fh.write(json.dumps(record, ensure_ascii=True) + "\n")
        self._meta_fh.flush()

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
        total_start = time.time()
        if robot_id is not None:
            self.robot_id = robot_id
            self._robot_config = get_robot_action_config(robot_id)
        concat_start = time.time()
        top = self._concat_cams_top(frames_by_cam)
        if self.video_scale != 1.0:
            h, w = top.shape[:2]
            new_w = max(2, int(w * self.video_scale))
            new_h = max(2, int(h * self.video_scale))
            top = cv2.resize(top, (new_w, new_h), interpolation=cv2.INTER_AREA)
        top_h, top_w = top.shape[:2]
        concat_elapsed = time.time() - concat_start

        # initialize writer once
        self._ensure_writer(episode_id, top_h, top_w)

        # write expects BGR
        meta_start = time.time()
        self._append_step_metadata(
            episode_id=episode_id,
            action=action,
            state=state,
            frames_by_cam=frames_by_cam,
        )
        meta_elapsed = time.time() - meta_start
        final_bgr = cv2.cvtColor(top, cv2.COLOR_RGB2BGR)

        if self._writer is None:
            raise RuntimeError("VideoWriter not initialized (unexpected).")
        write_start = time.time()
        self._writer.write(final_bgr)

        if (
            self.frame_save_interval > 0
            and (self._t + 1) % self.frame_save_interval == 0
            and self._frame_dir is not None
        ):
            frame_path = os.path.join(self._frame_dir, f"frame_{self._t + 1:06d}.png")
            cv2.imwrite(frame_path, final_bgr)
        write_elapsed = time.time() - write_start

        self._t += 1
        total_elapsed = time.time() - total_start
        if total_elapsed > 0.1:
            print(
                "[RecorderTiming] "
                f"episode={episode_id} step={self._t} "
                f"concat={concat_elapsed:.3f}s "
                f"meta={meta_elapsed:.3f}s "
                f"write={write_elapsed:.3f}s "
                f"total={total_elapsed:.3f}s"
            )

    def close(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._meta_fh is not None:
            self._meta_fh.close()
            self._meta_fh = None
        self._t = 0
        self._frame_w = None
        self._frame_h = None
        self._episode_dir = None
        self._frame_dir = None
        self._current_episode = None
