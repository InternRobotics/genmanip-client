from dis import Instruction
import time
import requests
import argparse
import pickle
import base64
import numpy as np
import torch
from PIL import Image
import io
import json_numpy
from tqdm import tqdm

from standalone_tools.client import EvalClient

from internmanip_client import AgentCfg, ServerCfg, AgentClient


class Gr00tClient:
    def __init__(self, model_host="0.0.0.0", model_port="8000"):
        server_cfg = ServerCfg(host=model_host, port=model_port)

        agent_cfg = AgentCfg(
            agent_type="gr00t_n1_5_arx_lift_bench",
            model_type="gr00t_n1_5",
            model_path="/mnt/workspace/work_projects/internmanip/Checkpoints/runs/gr00t_n1_5_internbench_pnp_base_bs24/checkpoint-10000",
            model_kwargs={"HF_cache_dir": None, "torch_dtype": "float16"},
            agent_settings={
                "data_config": "internbench",
                "embodiment_tag": "new_embodiment",
                "pred_action_horizon": 16,
                "action_ensemble": False,
                "adaptive_ensemble_alpha": 0,
            },
        )

        self.agent = AgentClient(agent_cfg, server_cfg)

    def get_action(self, obs):
        if obs["0"]["obs"]["reset"]:
            self.agent.reset()

        action = self.agent.step(obs)
        # convert output action
        action = {"0": action[0]}
        return action


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
    parser.add_argument("-cfg", "--config", type=str, default="")
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

    gr00t_client = Gr00tClient(args.model_host, args.model_port)
    print(f"Connect to model server")

    # wrap the eval loop in a try-finally to ensure cleanup
    try:

        for i in tqdm(range(1, 11)):
            if i in (6, 8, 9):
                continue
            client.load_config(f"configs/tasks/ebench/simple_pnp/task{i}.yml")

            obs = client.reset()

            while True:
                action = gr00t_client.get_action(obs)

                start = time.time()
                obs, done = client.step(action)
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
