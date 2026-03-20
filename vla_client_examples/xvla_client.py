import time
import requests
import argparse
import numpy as np
import json_numpy
from tqdm import tqdm

from genmanip_client import EvalClient


class XvlaClient:
    def __init__(
        self,
        server_url="http://0.0.0.0:8000",
        timeout=50,
        is_wheel_motion=True,
        gripper_value_type="discrete",
    ):
        self.server_url = server_url
        self.timeout = timeout
        self.is_wheel_motion = is_wheel_motion
        self.gripper_value_type = gripper_value_type

    def get_action(self, obs):
        instruction = obs["0"]["obs"]["instruction"]

        proprio = np.zeros(20, dtype=np.float32)
        proprio[:12] = obs["0"]["obs"]["state.joints"]

        payload = {
            "proprio": json_numpy.dumps(proprio),
            "language_instruction": instruction,
            "image0": json_numpy.dumps(
                obs["0"]["obs"]["video.top_camera_view"],
            ),
            "image1": json_numpy.dumps(
                obs["0"]["obs"]["video.left_camera_view"],
            ),
            "image2": json_numpy.dumps(
                obs["0"]["obs"]["video.right_camera_view"],
            ),
            "domain_id": 0,
            "steps": 10,
        }

        try:
            response = requests.post(
                f"{self.server_url}/act", json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            actions = np.array(result["action"], dtype=np.float32)
            # print(f"✅ Received {actions.shape} predicted actions.")
        except (
            requests.RequestException,
            KeyError,
            ValueError,
            TypeError,
            IndexError,
        ) as e:
            print(f"⚠️ Request failed: {e}")
            actions = np.zeros((30, 20), dtype=np.float32)

        # add delta action to current joint position
        action_list = []
        for i in range(30):
            action_list.append(
                self.prep_action(obs["0"]["obs"]["state.joints"], actions[i])
            )

        return action_list

    def prep_action(self, joints, action, control_type="joint_position"):
        gripper = action[12:14]
        base_motion = action[14:17] if self.is_wheel_motion else [0.0, 0.0, 0.0]

        if self.gripper_value_type == "discrete":
            gripper_joint = np.zeros(4)
            if gripper[0] < 0:
                gripper_joint[0:2] = np.array([0.044, 0.044])
            if gripper[1] < 0:
                gripper_joint[2:4] = np.array([0.044, 0.044])
        else:
            l_gripper = 0.0 if gripper[0] < 0.01 else gripper[0]
            r_gripper = 0.0 if gripper[1] < 0.01 else gripper[1]
            gripper_joint = [l_gripper, l_gripper, r_gripper, r_gripper]

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worker_ids",
        type=lambda s: s.split(","),
        default=["0"],
        help="List of worker IDs, i.e. --worker_ids 0,1,2",
    )
    parser.add_argument("--eval_host", type=str, default="0.0.0.0")
    parser.add_argument("--eval_port", type=int, default=8087)
    parser.add_argument("--model_host", type=str, default="0.0.0.0")
    parser.add_argument("--model_port", type=int, default=8000)
    parser.add_argument("-cfg", "--config", type=str, default="")
    parser.add_argument("-a", "--arm_type", type=str, default="r5a")
    parser.add_argument("-g", "--gripper_type", type=str, default="lift2")
    parser.add_argument("-c", "--control_type", type=str, default="joint_position")
    parser.add_argument("--disable_wheel_motion", action="store_true")
    parser.add_argument("--binary_gripper_value", action="store_false")
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

    model_url = f"http://{args.model_host}:{args.model_port}"
    xvla_client = XvlaClient(
        server_url=model_url,
        is_wheel_motion=not args.disable_wheel_motion,
        gripper_value_type="discrete" if args.binary_gripper_value else "continuous",
    )
    print(f"Connect to model server")

    # wrap the eval loop in a try-finally to ensure cleanup
    try:
        for i in tqdm(range(2, 3)):
            if i in (6, 8, 9):
                continue
            # client.load_config(f"configs/tasks/ebench/simple_pnp/task{i}.yml")

            done = False
            obs = client.reset()
            # {'0': {'obs': {'camera_data': {'left_camera': {'rgb': {480,640,3}, 'p', 'q'}}}}}
            #                'instruction': ...
            #                'joint_position_state': (16,)
            #                'ee_pose_state':[[p,q], [p,q]]

            while True:
                action = xvla_client.get_action(obs)

                # xvla return 30 actions, we apply first 15 actions
                for j in range(15):
                    cur_action = {i: action[j] for i in worker_ids}

                    # start = time.time()
                    obs, done = client.step(cur_action)
                    # print(f"workers {worker_ids} Step time: {time.time() - start:.4f} seconds")

                    if obs[worker_ids[0]]["obs"] is None or obs[worker_ids[0]]["obs"]["reset"]:  # type: ignore
                        # print(f"Episode {obs[worker_ids[0]]['obs']['episode_id']} started.")
                        break

                if done:
                    # finished all evaluations
                    break

    finally:
        client.kill_workers()
        print("Client cleaned.")
