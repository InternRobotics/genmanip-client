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
- `gmp clean`: Clean generated mesh cache, eval results, logs, and recursive lock/tmp leftovers.
- `gmp online create`: Create an online evaluation task.
- `gmp visualize`: Browse eval results and replay episodes in the Rerun viewer.
- `gmp online ready`: Check if an online evaluation task is ready.
- `gmp online submit`: Create an online evaluation task and poll until ready.

### gmp Examples

```bash
# Submit evaluation jobs
gmp submit configs/tasks/xxx.yml --host 127.0.0.1 --port 8087

# Run eval client
gmp eval --worker_ids 0,1 --host 127.0.0.1 --port 8087

# Preview cleanup targets
gmp clean --dry-run

# Remove generated mesh cache, eval results, logs, and lock/tmp files
gmp clean

# Also remove downloaded benchmark package cache
gmp clean --all

# Run eval client and auto-generate plots after each finished episode
gmp eval --worker_ids 0,1 --host 127.0.0.1 --port 8087 --plot_on_episode_end

# Generate plots for an existing episode directory
gmp plot client_results/<benchmark>/<run_id>/<task>/<seed>

# Visualize eval results from the current directory
gmp visualize

# Visualize with an explicit project root on a custom port
gmp visualize --project_root /path/to/GenManip-Sim --port 55088

# Show cached .rrd files that would be removed, then remove them
gmp visualize --flush-cache --dry-run
gmp visualize --flush-cache

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


## Episode Visualizer

`gmp visualize` starts a local HTTPS web server that lets you browse eval runs,
inspect per-task success rates, and replay individual episodes in the
[Rerun](https://rerun.io) WASM viewer — all from a normal browser with no
extra software required.

### Prerequisites

`rerun-sdk` is an optional dependency declared in `pyproject.toml`.
Install it via the `visualize` extra to get the pinned version:

```bash
pip install -e ".[visualize]"
```

The first launch downloads and caches the Rerun WASM viewer assets (~60 MB).
Subsequent launches reuse the cache and start in seconds.

### Basic usage

```bash
# From the project root (looks for saved/eval_results/ in cwd)
gmp visualize

# Explicit project root
gmp visualize --project_root /path/to/GenManip-Sim

# Custom port (default: 55077)
gmp visualize --port 55088
```

Then open the printed URL in your browser, e.g.:

```
https://<host>:55077/
```

> **Self-signed certificate** — the server uses HTTPS (required for the
> WebCodecs API that decodes MP4 video inside Rerun). On first visit the
> browser will show a certificate warning; click **Advanced → Proceed** to
> accept it. After that, all `fetch()` calls to the same origin work normally.

### Navigation

| Page | How to reach it | What you see |
|------|----------------|--------------|
| Home | `/` | All runs, sortable by name or time |
| Run detail | click a run | Task cards with per-task SR; episode dots (green = success, red = fail) |
| Episode viewer | click an episode dot | Rerun viewer with video + joint/gripper/base state & action curves |

Clicking an episode dot triggers a background `.rrd` build (first time only)
and reloads the Rerun iframe with the new recording. Subsequent clicks on the
same episode load instantly from the on-disk cache.

### Cache management

Each episode's `.rrd` file is cached as `.genmanip_vis.rrd` **inside the
episode directory** (next to the `.mp4` and `.pkl` files), so it lives on the
same storage mount as the raw data.

To inspect or remove all cached files:

```bash
# Show what would be deleted (dry run)
gmp visualize --flush-cache --dry-run

# Delete all cached .rrd files
gmp visualize --flush-cache
```

### Remote server workflow

If eval results live on a remote machine accessible via SSH port-forward:

```bash
# On your local machine — forward remote port 55077 to localhost
ssh -L 55077:localhost:55077 maker

# On the remote machine
gmp visualize --project_root /mnt/workspace/projects/GenManip-Sim

# Then open https://localhost:55077/ in your local browser
```

## Cleanup

`gmp clean` is intended to remove hard-to-notice runtime byproducts rather than user-facing client outputs.

By default it removes:

- `saved/assets/mesh_data`
- `saved/eval_results`
- `logs`
- Recursive `*.lock`, `*_soft.lock`, `*.tmp`, and `*.tmp-*` files under the workspace

By default it keeps:

- `client_results`
- `saved/tasks`
- `saved/demonstrations`

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
