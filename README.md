# genmanip-client

Client utilities for connecting to the GenManip eval server.

## Install (editable)

```bash
cd path/to/genmanip_client
pip install -e .
```

Optional deps for decoding tensors/images/numpy arrays:

```bash
pip install -e ".[full_numpy1]"
# OR
pip install -e ".[full_numpy2]"
```

## Usage

```bash
genmanip-client --host 127.0.0.1 --port 8087 --worker_ids 0,1
```

### CLI Arguments

- `--worker_ids`: Comma-separated worker IDs to attach to (default: `0`). Example: `--worker_ids 0,1,2`.
- `-cfg`, `--config`: Comma-separated config paths for `start_new_job` (default: `None`). Example: `--config configs/tasks/ebench/long/microwave.yml,configs/tasks/ebench/long/dishwasher.yml`.
- `--master`: Run as master client and call `start_new_job` before evaluation (default: `false`).
- `--run_id`: Run identifier passed to `start_new_job` (default: empty string).
- `--host`: Eval server host (default: `0.0.0.0`).
- `--port`: Eval server port (default: `8087`).
- `--reset`: Reset workers before stepping (default: `false`).
- `-a`, `--arm_type`: Arm type for generating fake actions (default: `franka`).
- `-g`, `--gripper_type`: Gripper type for generating fake actions (default: `panda_hand`).
- `-c`, `--control_type`: Control mode for generating fake actions (default: `joint_position`).
- `--robot_id`: Robot ID for action visualization; must be one of `ROBOT_ACTION_CONFIGS` (default: `None`).
