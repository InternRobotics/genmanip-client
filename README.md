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

Primary CLI (recommended):

```bash
gmp --help
```

```bash
genmanip-client --host 127.0.0.1 --port 8087 --worker_ids 0,1
```

### gmp Commands

- `gmp submit <config_paths...>`: Submit evaluation jobs to the server.
- `gmp eval`: Run the eval client (legacy behavior of `genmanip-client`).
- `gmp status`: Get current job status from the server.
- `gmp online create`: Create an online evaluation task.
- `gmp online ready`: Check if an online evaluation task is ready.
- `gmp online submit`: Create an online evaluation task and poll until ready.
- `gmp leaderboard list`: List local results under `saved/eval_results`.
- `gmp leaderboard submit`: Zip and submit a local result to the leaderboard.

### gmp Examples

```bash
# Submit evaluation jobs
gmp submit configs/tasks/xxx.yml --host 127.0.0.1 --port 8087

# Run eval client
gmp eval --worker_ids 0,1 --host 127.0.0.1 --port 8087

# Online evaluation: create and wait for endpoint, then eval
GMP_ONLINE_URL=$(gmp online submit --base_url https://example.com --token YOUR_TOKEN --task_id T2025123100001 --model_name internVLA --model_type VLA --benchmark_set EBench --print_endpoint)
gmp eval --url "$GMP_ONLINE_URL" --token YOUR_TOKEN

# Leaderboard: list and submit
gmp leaderboard list --project_root /path/to/GenManip-Sim
gmp leaderboard submit --run_id RUN_ID -n "My Submission" -l "EBench" --project_root /path/to/GenManip-Sim --host localhost --port 8000 --user_token YOUR_TOKEN
```

## 🖥️ Web Viewer (Headless-Friendly)

The GenManip client includes a lightweight web viewer for live camera streams when running in GUI-less terminals (e.g., DSW/SSH).

Start the viewer:

```bash
gmp eval --web_view
```

Open in a browser:

```
http://<machine-ip>:55090/
```

Optional flags:
- `--web_view_port port` to set a port for web viewer display
- `--web_view_interval N` to show one frame every N steps (default: 10)
- `--web_view_scale S` to scale the preview (default: 1.0)

### genmanip-client (legacy) Arguments

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
