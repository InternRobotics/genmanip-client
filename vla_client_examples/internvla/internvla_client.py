import time
import requests
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf

from standalone_tools.client import EvalClient
from models.internvla.websocket_policy_client import WebsocketClientPolicy


def read_mode_config(pretrained_checkpoint):
    if "mp_rank_00_model_states" in str(pretrained_checkpoint):
        run_dir = checkpoint_pt.parents[3]
    else:
        run_dir = checkpoint_pt.parents[1]

    config_yaml, dataset_statistics_json = (
        run_dir / "config.yaml",
        run_dir / "dataset_statistics.json",
    )

    ocfg = OmegaConf.load(str(config_yaml))
    global_cfg = OmegaConf.to_container(ocfg, resolve=True)

    with open(dataset_statistics_json, "r") as f:
        norm_stats = json.load(f)


class InternVLAClient:
    def __init__(
        self,
        server_url="0.0.0.0",
        port=10095,
        unnorm_key=None,
        policy_ckpt_path="",
        use_q99_normalization=True,
    ):
        self.unnorm_key = unnorm_key
        self.action_norm_stats, self.state_norm_stats = self.get_norm_stats(
            self.unnorm_key, policy_ckpt_path=policy_ckpt_path
        )
        self.use_q99_normalization = use_q99_normalization
        self.client = WebsocketClientPolicy(server)

    def get_norm_stats(self, unnorm_key, policy_ckpt_path):
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)
        if unnorm_key is None:
            assert (
                len(norm_stats) == 1
            ), "You train model with multi embodiments, please specify unnorm key!!"
            unnorm_key = next(iter(norm_stats.keys()))

        return norm_stats[unnorm_key]["action"], norm_stats[unnorm_key]["state"]

    def intput_wrapper(self, obs):
        # 1. Normalization state
        if self.use_q99_normalization:
            state_low, state_high = np.array(self.state_norm_stats["q01"]), np.array(
                self.state_norm_stats["q99"]
            )
        else:
            state_low, state_high = np.array(self.state_norm_stats["min"]), np.array(
                self.state_norm_stats["max"]
            )
        intput_state = obs["0"]["obs"]["joint_position_state"]
        norm_joint_left = (
            2
            * (intput_state[0:6] - state_low[:6])
            / (state_high[:6] - state_low[:6] + 1e-8)
            - 1
        )
        norm_joint_right = (
            2
            * (intput_state[8:14] - state_low[8:14])
            / (state_high[6:12] - state_low[6:12] + 1e-8)
            - 1
        )

        FULL_STATE_DIM = 36
        norm_state = -np.ones(FULL_STATE_DIM)
        norm_state[0:6] = norm_joint_left
        norm_state[17:23] = norm_joint_right

        obs["0"]["obs"]["norm_joint_position"] = norm_state
        return obs

    def output_wrapper(self, normalized_actions):
        # 1. Unnormalize action
        if self.use_q99_normalization:
            action_high, action_low = np.array(action_norm_stats["q99"]), np.array(
                action_norm_stats["q01"]
            )
        else:
            action_high, action_low = np.array(action_norm_stats["max"]), np.array(
                action_norm_stats["min"]
            )
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_delta_position_left = normalized_actions[:, 0:6]
        normalized_absolute_gripper_left = normalized_actions[:, 6:8]
        normalized_delta_position_right = normalized_actions[:, 17:23]
        normalized_absolute_gripper_right = normalized_actions[:, 23:25]

        unnorm_delta_position_left = (
            0.5
            * (normalized_delta_position_left + 1)
            * (action_high[0:6] - action_low[0:6])
            + action_low[0:6]
        )
        unnorm_delta_position_right = (
            0.5
            * (normalized_delta_position_right + 1)
            * (action_high[6:12] - action_low[6:12])
            + action_low[6:12]
        )

        return np.concatenate(
            [
                unnorm_delta_position_left,
                normalized_absolute_gripper_left,
                unnorm_delta_position_right,
                normalized_absolute_gripper_right,
            ],
            axis=1,
        )

    def get_action(self, obs):
        obs = self.intput_wrapper(obs)
        camera_keys = ["top_camera", "left_camera", "right_camera"]
        images = [obs["0"]["obs"]["camera_data"][key]["rgb"] for key in camera_keys]
        instruction = obs["0"]["obs"]["instruction"]
        state = obs["0"]["obs"]["norm_joint_position"]

        payload = {
            "batch_images": [images],
            "state": [
                [state]
            ],  # TODO(yuqiang yang) Remove the duplicated [] after fixing the training problem
            "instructions": [instruction],
        }

        response = self.client.infer(payload)

        action_list = self.output_wrapper(response["data"]["normalized_actions"][0])
        # add delta action to current joint position
        action_list = []
        for i in range(30):
            action_list.append(self.prep_action(proprio[:12], actions[i]))

        return action_list

    def prep_action(self, joints, action, control_type="joint_position"):
        gripper = action[12:14]
        base_motion = action[14:17]
        gripper_joint = np.zeros(4)
        if gripper[0] <= 0.9:
            gripper_joint[0:2] = np.array([0.044, 0.044])
        if gripper[1] <= 0.9:
            gripper_joint[2:4] = np.array([0.044, 0.044])
        joint = np.concatenate(
            [
                action[:6] + joints[:6],  # joints 0-5
                gripper_joint[0:2],  # left gripper
                action[6:12] + joints[6:12],  # joints 6-11
                gripper_joint[2:4],  # right gripper
            ]
        )
        # print(f"joint: {joint}, base_motion: {base_motion}")
        return {
            "action": joint,
            "base_motion": base_motion,
            "control_type": control_type,
        }


def parse_list(s):
    return s.split(",")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worker_ids",
        type=parse_list,
        default=["0"],
        help="List of worker IDs, i.e. --worker_ids 0,1,2",
    )
    parser.add_argument("--eval_host", type=str, default="0.0.0.0")
    parser.add_argument("--eval_port", type=int, default=8087)
    parser.add_argument("--model_host", type=str, default="0.0.0.0")
    parser.add_argument("--model_port", type=int, default=8000)
    parser.add_argument("--pretrained_path", type=str)
    parser.add_argument("--unnorm_key", type=str, default=None)
    parser.add_argument("-cfg", "--config", type=str, default="")
    parser.add_argument("-a", "--arm_type", type=str, default="r5a")
    parser.add_argument("-g", "--gripper_type", type=str, default="lift2")
    parser.add_argument("-c", "--control_type", type=str, default="joint_position")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    host = args.eval_host
    port = args.eval_port
    worker_ids = args.worker_ids
    base_url = f"http://{host}:{port}"

    # Create workers on server here, make sure they are created before stepping
    client = EvalClient(base_url, worker_ids, args.config)
    print(f"Created workers {worker_ids} on server {base_url}.")

    xvla_client = InternVLAClient(
        server_url=args.model_host,
        port=args.model_port,
        policy_ckpt_path=args.pretrained_path,
        use_q99_normalization=True,
    )
    print(f"Connect to model server")

    # wrap the eval loop in a try-finally to ensure cleanup
    try:

        for i in tqdm(range(1, 11)):
            if i in (6, 8, 9):
                continue
            print(f"loading config configs/tasks/ebench/simple_pnp/task{i}.yml")
            client.load_config(f"configs/tasks/ebench/simple_pnp/task{i}.yml")

            obs = client.reset()
            # {'0': {'obs': {'camera_data': {'left_camera': {'rgb': {480,640,3}, 'p', 'q'}}}}}
            #                'instruction': ...
            #                'joint_position_state': (16,)
            #                'ee_pose_state':[[p,q], [p,q]]

            while True:

                action = xvla_client.get_action(obs)
                # xvla return 30 actions, we apply first 15 actions

                start = time.time()
                for j in range(15):
                    cur_action = {i: action[j] for i in worker_ids}

                    obs, done = client.step(cur_action)
                # print(f"workers {worker_ids} Step time: {time.time() - start:.4f} seconds")

                if done:
                    # finished all evaluations
                    break
                if obs is None:
                    break
                if obs[worker_ids[0]]["obs"]["reset"]:  # type: ignore
                    # model.reset()
                    pass
    finally:
        client.kill_workers()
        print("Client cleaned.")
