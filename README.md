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

### gmp Commands

- `gmp submit <config_paths...>`: Submit evaluation jobs to the server.
- `gmp eval`: Run the eval client (legacy behavior of `genmanip-client`).
- `gmp plot <episode_dir>`: Generate action/state plots and a merged plot video for one episode.
- `gmp status`: Get current job status from the server.
- `gmp online create`: Create an online evaluation task.
- `gmp online ready`: Check if an online evaluation task is ready.
- `gmp online submit`: Create an online evaluation task and poll until ready.

### gmp Examples

```bash
# Submit evaluation jobs
gmp submit configs/tasks/xxx.yml --host 127.0.0.1 --port 8087

# Run eval client
gmp eval --worker_ids 0,1 --host 127.0.0.1 --port 8087

# Run eval client and auto-generate plots after each finished episode
gmp eval --worker_ids 0,1 --host 127.0.0.1 --port 8087 --plot_on_episode_end

# Generate plots for an existing episode directory
gmp plot client_results/<benchmark>/<run_id>/<task>/<seed>

# Online evaluation: create and wait for endpoint, then eval
resp=$(gmp online submit --base_url https://example.com --token YOUR_TOKEN --task_id T2025123100001 --model_name internVLA --model_type VLA --benchmark_set EBench --print_endpoint)
GMP_ONLINE_URL=$(printf '%s' "$resp" | jq -r '.endpoint')
TASK_ID=$(printf '%s' "$resp" | jq -r '.task_id')
gmp eval --url "$GMP_ONLINE_URL" --run_id "$TASK_ID" --token YOUR_TOKEN

# Leaderboard: list and submit
gmp leaderboard list --project_root /path/to/GenManip-Sim
gmp leaderboard submit --run_id RUN_ID -n "My Submission" -l "EBench" --project_root /path/to/GenManip-Sim --host localhost --port 8000 --user_token YOUR_TOKEN
```

## Online Evaluation Workflow

The current online flow is intended for users who run their VLA model locally and connect it to a remote GenManip evaluation service.

### 1. Register on the platform

Create an account on the web platform first, then obtain:

- `base_url`: the online evaluation service URL
- `token`: your API token

### 2. Install the client

```bash
cd path/to/genmanip_client
pip install -e .
```

If you want to parse the JSON result from `gmp online submit --print_endpoint` in shell, also install `jq`.

### 3. Submit an online evaluation task

`gmp online submit` creates the task and keeps polling until compute resources are assigned and the eval endpoint is ready.

```bash
gmp online submit \
  --base_url https://example.com \
  --token YOUR_TOKEN \
  --task_id T2025123100001 \
  --model_name internVLA \
  --model_type VLA \
  --benchmark_set EBench
```

Typical ready response:

```json
{
  "task_id": "T2025123100001",
  "endpoint": "https://example.com/eval/..."
}
```

Notes:

- `gmp online submit` waits and polls by default until the endpoint is ready.
- Use `--print_endpoint` if you only want machine-readable output for shell capture.
- `task_id` can be reused as the `run_id` of the eval client so the online platform and local eval run stay aligned.

### 4. Start local evaluation with the returned endpoint

Recommended shell workflow:

```bash
resp=$(gmp online submit \
  --base_url https://example.com \
  --token YOUR_TOKEN \
  --task_id T2025123100001 \
  --model_name internVLA \
  --model_type VLA \
  --benchmark_set EBench \
  --print_endpoint)

GMP_ONLINE_URL=$(printf '%s' "$resp" | jq -r '.endpoint')
TASK_ID=$(printf '%s' "$resp" | jq -r '.task_id')

gmp eval \
  --url "$GMP_ONLINE_URL" \
  --run_id "$TASK_ID" \
  --token YOUR_TOKEN
```

If your policy supports action chunking, you can reduce request overhead with:

```bash
gmp eval \
  --url "$GMP_ONLINE_URL" \
  --run_id "$TASK_ID" \
  --token YOUR_TOKEN \
  --chunk_size 8
```

### Service-side note

When launching the internal evaluation server, disable episode image dumping to avoid unnecessary overhead:

```bash
python ray_eval_server.py --episode_recorder_save_every 0
```

This is especially useful for online evaluation where saving images is usually not needed.

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

## Plotting Episode Results

Use `gmp plot` to post-process one saved episode directory:

```bash
gmp plot client_results/<benchmark>/<run_id>/<task>/<seed>
```

It reads:

- `steps.jsonl` for action/state traces
- `merged*.mp4` for the recorded client video

And writes:

- `action_plot.png`
- `state_plot.png`
- `merged_with_plot*.mp4`

If `ffmpeg` is available, the final plot video is transcoded to H.264 with `yuv420p` and `faststart` for better playback compatibility.

To generate these plots automatically at episode end during evaluation:

```bash
gmp eval --plot_on_episode_end
```

This launches `gmp plot` asynchronously after each finished episode and writes logs to `plot.log` inside the episode directory.
