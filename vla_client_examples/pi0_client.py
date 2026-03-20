import time
import argparse
from tqdm import tqdm
import numpy as np

from openpi_client import image_tools
from openpi_client import websocket_client_policy

from genmanip_client import EvalClient


class Pi0Client:
    """
    Run under conda env client
    """

    def __init__(self, model_host="0.0.0.0", model_port="8000"):
        self.client = websocket_client_policy.WebsocketClientPolicy(
            host=model_host, port=8000
        )

    def get_action(self, obs):
        obs = self.prep_input(obs)
        action_chunk = self.client.infer(obs)["actions"]
        actions = self.prep_output(action_chunk)
        return actions

    def prep_input(self, obs):
        observation = {
            "images/head": obs["0"]["obs"]["video.top_camera_view"],
            "images/hand_left": obs["0"]["obs"]["video.left_camera_view"],
            "images/hand_right": obs["0"]["obs"]["video.right_camera_view"],
            "states/joint": obs["0"]["obs"]["state.joints"],
            "states/gripper": obs["0"]["obs"]["state.gripper"],
            "states/base": obs["0"]["obs"]["state.base"],
            "prompt": obs["0"]["obs"]["instruction"],
        }
        return observation

    def prep_output(self, action_chunk):
        actions = []
        for i in range(10):
            action = action_chunk[i]
            gripper = action[12:14]
            base_motion = action[14:17]
            gripper_joint = np.zeros(4)
            if gripper[0] <= 0.0:
                gripper_joint[0:2] = np.array([0.044, 0.044])  # open
            if gripper[1] <= 0.0:
                gripper_joint[2:4] = np.array([0.044, 0.044])  # open

            joint = np.concatenate(
                [
                    action[:6],  # joints 0-5
                    gripper_joint[0:2],  # left gripper
                    action[6:12],  # joints 6-11
                    gripper_joint[2:4],  # right gripper
                ]
            )
            action_dict = {
                "action": joint,
                "base_motion": base_motion,
                "control_type": "joint_position",
            }
            actions.append(action_dict)
        return actions


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
    parser.add_argument("--model_host", type=str, default="10.150.129.169")
    parser.add_argument("--model_port", type=int, default=8000)
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--reset", action="store_true")
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
    config = args.config

    # Create workers on server here, make sure they are created before stepping
    client = EvalClient(base_url, worker_ids, config)
    print(f"Created workers {worker_ids} on server {base_url}.")

    pi0_client = Pi0Client(args.model_host, args.model_port)
    print(f"Connect to model server")

    # wrap the eval loop in a try-finally to ensure cleanup
    try:

        for i in tqdm(range(1, 11)):
            if i in (6, 8, 9):
                continue
            client.load_config(f"configs/tasks/ebench/simple_pnp/task{i}.yml")

            obs = client.reset()

            while True:
                actions = pi0_client.get_action(obs)

                start = time.time()
                for action in actions:
                    obs, done = client.step({"0": action})
                    if done or obs[worker_ids[0]]["obs"]["reset"]:
                        break

                # print(f"workers {worker_ids} Step time: {time.time() - start:.4f} seconds")

                if done:
                    result_metric = obs[worker_ids[0]][
                        "metric"
                    ]  # finished all evaluations
                    # print(result_metric)
                    break

                if obs is None:
                    break

    finally:
        client.kill_workers()
        print("Client cleaned.")
