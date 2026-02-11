import argparse
import base64
from filelock import SoftFileLock
import io
import json
import multiprocessing
import os
import threading
import sys
from functools import wraps
from pathlib import Path
import pickle
import tempfile
import time
from typing import Any
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from importlib import import_module
import numpy as np
import requests

from .vis_utils import ROBOT_ACTION_CONFIGS, StreamingEpisodeRecorder, concat_cams_top


# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_CYAN = "\033[96m"


class Box:
    TL = "╭"
    TR = "╮"
    BL = "╰"
    BR = "╯"
    H = "─"
    V = "│"
    LT = "├"
    RT = "┤"


def colored(text: str, *colors: str) -> str:
    """Wrap text with ANSI color codes."""
    if not sys.stdout.isatty():
        return text
    return "".join(colors) + text + Colors.RESET


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _visible_len(text: str) -> int:
    return len(_ANSI_RE.sub("", text))


def make_box_line(
    content: str,
    width: int,
    left: str = Box.V,
    right: str = Box.V,
    border_color: str | None = None,
) -> str:
    """Create a box line with content padded to width (ANSI-aware)."""
    padding = width - _visible_len(content)
    left_border = colored(left, border_color) if border_color else left
    right_border = colored(right, border_color) if border_color else right
    return f"{left_border} {content}{' ' * max(0, padding)} {right_border}"


def print_info(message: str) -> None:
    icon = colored("→", Colors.CYAN)
    print(f"{icon} {message}")


def print_success(message: str) -> None:
    icon = colored("✓", Colors.BRIGHT_GREEN, Colors.BOLD)
    print(f"{icon} {colored(message, Colors.BRIGHT_GREEN)}")


# Timeout constants (in seconds)
DEFAULT_STEP_TIMEOUT = 600  # 10 minutes
DEFAULT_RESET_TIMEOUT = 600  # 10 minutes
DEFAULT_CREATE_TIMEOUT = 600  # 10 minutes
DEFAULT_HEALTH_CHECK_TIMEOUT = 5.0


def _retry_on_failure(max_retries: int = 3, backoff: float = 1.0):
    """Decorator that retries a function on connection/timeout failures."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (requests.ConnectionError, requests.Timeout) as e:
                    last_exc = e
                    if attempt < max_retries - 1:
                        time.sleep(backoff * (2**attempt))
            raise RuntimeError(
                f"Failed after {max_retries} attempts: {last_exc}"
            ) from last_exc

        return wrapper

    return decorator


def _optional_import(name: str):
    try:
        module = import_module(name)
    except Exception as exc:
        raise RuntimeError(
            f"Missing optional dependency '{name}'. "
            f"Install with: pip install -e '.[full]' (or install '{name}' directly)."
        ) from exc
    return module


def decode_numpy(metadata: dict) -> "Any":
    np = _optional_import("numpy")
    decoded_bytes = base64.b64decode(metadata["data"])
    numpy_array = np.frombuffer(decoded_bytes, dtype=np.dtype(metadata["dtype"]))
    numpy_array = numpy_array.reshape(metadata["shape"])
    return numpy_array


def decode_tensor(metadata: dict) -> "Any":
    torch = _optional_import("torch")
    decoded_bytes = base64.b64decode(metadata["data"])
    tensor = torch.frombuffer(
        bytearray(decoded_bytes), dtype=getattr(torch, metadata["dtype"])
    )
    tensor = tensor.reshape(eval(metadata["shape"]))
    return tensor.to(metadata["device"])


def decode_image(metadata: dict) -> "Any":
    try:
        Image = _optional_import("PIL.Image")
        decoded_bytes = base64.b64decode(metadata["data"])
        image = Image.open(io.BytesIO(decoded_bytes))

        if "size" in metadata and image.size != metadata["size"]:
            image = image.resize(metadata["size"], Image.Resampling.LANCZOS)

        if "mode" in metadata and image.mode != metadata["mode"]:
            image = image.convert(metadata["mode"])

        return image
    except Exception as exc:
        raise RuntimeError(f"Image decoding failed: {exc}") from exc


def deserialize_data(data: Any):
    if isinstance(data, dict) and "type" in data:
        if data["type"] == "numpy_array":
            return decode_numpy(data)
        if data["type"] == "tensor":
            return decode_tensor(data)
        if data["type"] == "image":
            return decode_image(data)
    if isinstance(data, (list, tuple)):
        return [deserialize_data(item) for item in data]
    if isinstance(data, dict):
        return {key: deserialize_data(value) for key, value in data.items()}
    return data


def _storage_worker_process(
    task_queue: Any,  # multiprocessing.Queue
    done_event: Any,  # multiprocessing.Event
    pending_counter: Any,  # multiprocessing.Value
    log_dir: str,
    worker_ids: list[str],
    fps: int,
    cam_order: list[str],
    robot_id: str | None,
) -> None:
    """
    Worker process function for async storage operations.
    Runs in a separate process to avoid GIL limitations.
    """
    # Import inside process to avoid pickling issues
    import numpy as np

    from .vis_utils import StreamingEpisodeRecorder

    # Create recorders inside the child process (cv2.VideoWriter is not picklable)
    recorders: dict[str, StreamingEpisodeRecorder] = {}
    for wid in worker_ids:
        recorders[wid] = StreamingEpisodeRecorder(
            out_dir=log_dir,
            fps=fps,
            plot_height=600,
            video_scale=0.75,
            cam_order=cam_order,
            robot_id=robot_id,
        )

    def process_record(task: dict) -> None:
        """Process a recording task."""
        obs = task["obs"]
        action_dict = task["action_dict"]

        # Attach action into obs dict for each worker
        for wid, act in action_dict.items():
            if wid in obs:
                obs[wid]["action"] = act

        # Streaming record
        for wid in worker_ids:
            wdata = obs.get(wid, {})
            if not wdata or wdata["obs"] is None or wdata["obs"]["reset"]:
                continue
            wobs = wdata.get("obs", {})
            episode_id = wobs.get("episode_id", "unknown_episode")
            # Collect camera frames from current obs
            frames_by_cam = {}
            for k, v in list(wobs.items()):
                if isinstance(k, str) and k.startswith("video."):
                    cam_name = k.split(".", 1)[1]
                    if isinstance(v, np.ndarray):
                        frames_by_cam[cam_name] = v

            if frames_by_cam:
                recorders[wid].write_step(
                    episode_id=episode_id,
                    frames_by_cam=frames_by_cam,
                    action=wdata.get("action", None),
                    robot_id=wobs.get("robot_id", None),
                )

    def save_episode_sr(episode_result: dict) -> None:
        """Save episode success rate to disk."""
        episode_id = episode_result.get("episode_id")
        task_name = episode_result.get("task_name")
        seed = episode_result.get("seed")
        sr = episode_result.get("sr")

        if episode_id is None or task_name is None or seed is None or sr is None:
            return

        try:
            sr_value = float(sr)
        except (TypeError, ValueError):
            return

        episode_dir = Path(log_dir) / str(episode_id)
        episode_dir.mkdir(parents=True, exist_ok=True)
        episode_sr_path = episode_dir / "sr.json"
        with episode_sr_path.open("w", encoding="utf-8") as f:
            json.dump({"sr": sr_value}, f, indent=2)

        task_dir = episode_dir.parent
        task_sr_path = task_dir / "episode_sr.json"
        lock_path = task_sr_path.with_suffix(".lock")

        with SoftFileLock(lock_path):
            task_sr = {}

            if task_sr_path.exists():
                with task_sr_path.open("r", encoding="utf-8") as f:
                    task_sr = json.load(f)

            task_sr[str(seed)] = sr_value

            dir_path = task_sr_path.parent
            fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(task_sr, f, indent=2, sort_keys=True)
                    f.flush()
                    os.fsync(f.fileno())

                os.replace(tmp_path, task_sr_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    # Main loop
    while not done_event.is_set():
        try:
            task = task_queue.get(timeout=0.1)
        except Exception:
            # Queue.Empty or other exceptions
            continue

        if task is None:
            # Poison pill received, exit the loop
            break

        task_type = task.get("type")
        try:
            if task_type == "record":
                process_record(task)
            elif task_type == "episode_result":
                save_episode_sr(task["episode_result"])
        except Exception as e:
            print(
                f"\033[31m✗\033[0m [StorageWorker] Error processing task {task_type}: {e}"
            )
        finally:
            # Decrement pending counter
            with pending_counter.get_lock():
                pending_counter.value -= 1

    # Close all recorders before exiting
    for r in recorders.values():
        r.close()


class StorageWorker:
    """
    Background process worker for async storage operations.
    Handles video recording and episode result saving without blocking the main loop.
    Uses multiprocessing to bypass GIL limitations for CPU-bound tasks.
    """

    def __init__(
        self,
        log_dir: str,
        worker_ids: list[str],
        fps: int,
        cam_order: list[str],
        robot_id: str | None,
    ):
        self.log_dir = log_dir
        self.worker_ids = worker_ids
        self.fps = fps
        self.cam_order = cam_order
        self.robot_id = robot_id

        # Use multiprocessing primitives
        self._queue = multiprocessing.Queue()
        self._done_event = multiprocessing.Event()
        self._pending_counter = multiprocessing.Value("i", 0)
        self._process: multiprocessing.Process | None = None

        self._start_process()

    def _start_process(self) -> None:
        self._process = multiprocessing.Process(
            target=_storage_worker_process,
            args=(
                self._queue,
                self._done_event,
                self._pending_counter,
                self.log_dir,
                self.worker_ids,
                self.fps,
                self.cam_order,
                self.robot_id,
            ),
            daemon=True,
        )
        self._process.start()

    def enqueue_record(self, obs: dict, action_dict: dict) -> None:
        """Enqueue a recording task (non-blocking)."""
        # Deep copy the data to avoid mutation issues
        obs_copy = self._deep_copy_obs(obs)
        action_copy = {k: v for k, v in action_dict.items()}
        # Increment pending counter before putting to queue
        with self._pending_counter.get_lock():
            self._pending_counter.value += 1
        self._queue.put(
            {
                "type": "record",
                "obs": obs_copy,
                "action_dict": action_copy,
            }
        )

    def enqueue_episode_result(self, episode_result: dict) -> None:
        """Enqueue an episode result saving task (non-blocking)."""
        # Increment pending counter before putting to queue
        with self._pending_counter.get_lock():
            self._pending_counter.value += 1
        self._queue.put(
            {
                "type": "episode_result",
                "episode_result": episode_result.copy(),
            }
        )

    def _deep_copy_obs(self, obs: dict) -> dict:
        """Deep copy observation dict, making copies of numpy arrays."""
        result = {}
        for wid, wdata in obs.items():
            if not isinstance(wdata, dict):
                result[wid] = wdata
                continue
            wdata_copy = {}
            for k, v in wdata.items():
                if k == "obs" and isinstance(v, dict):
                    obs_copy = {}
                    for ok, ov in v.items():
                        if isinstance(ov, np.ndarray):
                            obs_copy[ok] = ov.copy()
                        else:
                            obs_copy[ok] = ov
                    wdata_copy[k] = obs_copy
                elif isinstance(v, np.ndarray):
                    wdata_copy[k] = v.copy()
                else:
                    wdata_copy[k] = v
            result[wid] = wdata_copy
        return result

    def close(self) -> None:
        """Stop the worker process and wait for it to finish."""
        # Signal the process to stop
        self._done_event.set()
        # Send poison pill to ensure process exits
        self._queue.put(None)
        # Wait for process to finish
        if self._process is not None:
            self._process.join(timeout=30.0)
            if self._process.is_alive():
                print(
                    "\033[33m⚠\033[0m [StorageWorker] Process did not exit in time, terminating..."
                )
                self._process.terminate()
                self._process.join(timeout=5.0)

    def wait_until_done(self) -> None:
        """Wait until all queued tasks are processed."""
        while True:
            with self._pending_counter.get_lock():
                if self._pending_counter.value <= 0:
                    break
            time.sleep(0.05)


class EvalClient:
    """
    EvalClient in binary mode:
    - /step:  pickled action_dict <-> pickled response_dict
    - /reset: pickled {"worker_ids": [...]} <-> pickled obs_dict
    - /kill: kill all workers
    - /load_config: load a new task config, restart server
    The rest APIs are still in JSON.
    """

    def __init__(
        self,
        base_url: str,
        worker_ids: list[str] = ["0"],
        save_result: bool = True,
        fps: int = 30,
        cam_order: list[str] | None = None,
        robot_id: str | None = None,
        step_timeout: float = DEFAULT_STEP_TIMEOUT,
        reset_timeout: float = DEFAULT_RESET_TIMEOUT,
        run_id: str = "",
        verbose: bool = True,
        token: str | None = None,
        web_view: bool = False,
        web_view_port: int = 55090,
        web_view_interval: int = 10,
        web_view_scale: float = 1.0,
    ):
        # Print startup banner
        self.verbose = verbose
        inner_width = 46
        print(colored(f"{Box.TL}{Box.H * (inner_width+2)}{Box.TR}", Colors.CYAN))
        title = "GenManip Evaluation Client"
        title_pad = max(0, (inner_width - len(title)) // 2)
        title_line = (
            " " * title_pad
            + colored(title, Colors.BOLD, Colors.BRIGHT_CYAN)
            + " " * (inner_width - title_pad - len(title))
        )
        print(
            make_box_line(
                title_line,
                inner_width,
                border_color=Colors.CYAN,
            )
        )
        print(colored(f"{Box.BL}{Box.H * (inner_width+2)}{Box.BR}", Colors.CYAN))
        print()

        self.base_url = base_url.rstrip("/")
        self.step_timeout = step_timeout
        self.reset_timeout = reset_timeout
        self.run_id = run_id
        self.log_dir = os.environ.get("GENMANIP_RESULT_DIR", "client_results")
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        print_info(f"Results directory: {colored(self.log_dir, Colors.CYAN)}")

        self.worker_ids = worker_ids
        self.robot_id = robot_id
        self._auth_headers: dict[str, str] = {}
        if token:
            self._auth_headers = {"Authorization": f"Bearer {token}"}

        # Health check before operations
        self._health_check()

        self.save_result = save_result
        self._storage_worker: StorageWorker | None = None
        if self.save_result:
            self.fps = fps
            self.cam_order = cam_order or [
                "left_camera_view",
                "top_camera_view",
                "right_camera_view",
                "obs_camera_view",
                "realsense_view",
                "obs_camera_2_view",
            ]
            self._storage_worker = StorageWorker(
                log_dir=self.log_dir,
                worker_ids=self.worker_ids,
                fps=self.fps,
                cam_order=self.cam_order,
                robot_id=self.robot_id,
            )

        self.step_count = 0
        self._web_view = web_view
        self._web_view_port = int(web_view_port)
        self._web_view_interval = max(1, int(web_view_interval))
        self._web_view_scale = float(web_view_scale)
        self._web_server: ThreadingHTTPServer | None = None
        self._web_thread: threading.Thread | None = None
        self._web_frame_lock = threading.Lock()
        self._web_frame_jpeg: bytes | None = None
        self._web_cv2 = None
        if self._web_view:
            self._start_web_viewer()
        print_info(f"Connected to server: {colored(base_url, Colors.CYAN)}")
        print_info(f"Workers: {colored(str(self.worker_ids), Colors.YELLOW)}")
        print()

    def _start_web_viewer(self) -> None:
        try:
            import cv2  # type: ignore
        except Exception:
            print_info("Web viewer disabled (opencv-python not available).")
            self._web_view = False
            return
        self._web_cv2 = cv2

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path in ("/", "/index.html"):
                    body = (
                        "<html><head><title>GenManip Stream</title>"
                        "<style>body{font-family:Arial,Helvetica,sans-serif;background:#111;color:#ddd;text-align:center}"
                        "img{max-width:96vw;max-height:92vh;margin-top:10px;border:1px solid #333}</style>"
                        "</head><body><h3>GenManip Stream</h3>"
                        "<img src='/frame.jpg' id='f' />"
                        "<script>"
                        "setInterval(()=>{const img=document.getElementById('f');"
                        "img.src='/frame.jpg?t='+Date.now();}, 200);"
                        "</script></body></html>"
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path.startswith("/frame.jpg"):
                    with self.server._frame_lock:
                        data = self.server._frame_jpeg
                    if not data:
                        self.send_response(204)
                        self.end_headers()
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                self.send_response(404)
                self.end_headers()

            def log_message(self, format: str, *args: Any) -> None:
                return

        try:
            server = ThreadingHTTPServer(("0.0.0.0", self._web_view_port), _Handler)
        except OSError as exc:
            print_info(f"Web viewer disabled (port {self._web_view_port} unavailable): {exc}")
            self._web_view = False
            return
        server._frame_lock = self._web_frame_lock  # type: ignore[attr-defined]
        server._frame_jpeg = None  # type: ignore[attr-defined]
        self._web_server = server
        self._web_thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._web_thread.start()
        print_info(f"Web viewer: http://0.0.0.0:{self._web_view_port}")

    def _stop_web_viewer(self) -> None:
        if self._web_server is None:
            return
        try:
            self._web_server.shutdown()
            self._web_server.server_close()
        except Exception:
            pass
        self._web_server = None
        self._web_thread = None

    def _health_check(self, timeout: float = DEFAULT_HEALTH_CHECK_TIMEOUT):
        """Check server connectivity before operations."""
        try:
            resp = requests.get(
                f"{self.base_url}/docs",
                timeout=timeout,
                headers=self._build_headers(),
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Server health check failed with status {resp.status_code}"
                )
        except requests.RequestException as e:
            raise RuntimeError(
                f"Cannot connect to server at {self.base_url}. "
                f"Ensure the evaluation server is running. Error: {e}"
            ) from e

    # ================= Lifecycle =================
    def _build_headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = dict(self._auth_headers)
        if extra:
            headers.update(extra)
        return headers

    def close(self) -> None:
        """Close recorders."""
        self.close_recorders()
        self._stop_web_viewer()
        self.kill_workers()
        print()
        print_success(f"Client closed.")

    def close_recorders(self) -> None:
        """Close storage worker and wait for pending tasks to complete."""
        if self._storage_worker is not None:
            # Wait for all queued tasks to finish before closing
            self._storage_worker.wait_until_done()
            self._storage_worker.close()
            self._storage_worker = None

    def _record(self, obs: dict, action_dict: dict) -> None:
        """Enqueue recording task to background storage worker (non-blocking)."""
        if self._storage_worker is not None:
            self._storage_worker.enqueue_record(obs, action_dict)

    def _record_episode_results(self, obs: dict) -> None:
        """Enqueue episode result saving to background storage worker (non-blocking)."""
        if self._storage_worker is None:
            return
        for wdata in obs.values():
            episode_result = wdata.get("episode_result")
            if episode_result is not None and isinstance(episode_result, dict):
                self._storage_worker.enqueue_episode_result(episode_result)

    @_retry_on_failure(max_retries=3, backoff=1.0)
    def _create_workers(self):
        try:
            resp = requests.post(
                f"{self.base_url}/create_workers",
                json={"data": {"worker_ids": self.worker_ids}},
                timeout=DEFAULT_CREATE_TIMEOUT,
                headers=self._build_headers(),
            )
        except requests.Timeout:
            raise RuntimeError(
                f"Create workers request timed out after {DEFAULT_CREATE_TIMEOUT}s"
            )
        if resp.status_code != 200:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise RuntimeError(
                f"HTTP error when create_workers: {resp.status_code} - {detail}"
            )

    def reset(self):
        payload = pickle.dumps(
            {"worker_ids": self.worker_ids}, protocol=pickle.HIGHEST_PROTOCOL
        )
        try:
            resp = requests.post(
                f"{self.base_url}/reset",
                data=payload,
                headers=self._build_headers(
                    {"Content-Type": "application/octet-stream"}
                ),
                timeout=self.reset_timeout,
            )
        except requests.Timeout:
            raise RuntimeError(f"Reset request timed out after {self.reset_timeout}s")
        except requests.RequestException as exc:
            raise RuntimeError(f"HTTP request to server failed: {exc}") from exc

        if resp.status_code != 200:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise RuntimeError(f"HTTP error on reset: {resp.status_code} - {detail}")

        obs_dict = pickle.loads(resp.content)
        obs = deserialize_data(obs_dict)
        print_info("Environment reset complete")
        print()
        return obs

    def step(self, action_dict: dict) -> tuple[dict, bool]:
        start = time.time()
        payload = pickle.dumps(action_dict, protocol=pickle.HIGHEST_PROTOCOL)
        try:
            resp = requests.post(
                f"{self.base_url}/step",
                data=payload,
                headers=self._build_headers(
                    {"Content-Type": "application/octet-stream"}
                ),
                timeout=self.step_timeout,
            )
        except requests.Timeout:
            raise RuntimeError(f"Step request timed out after {self.step_timeout}s")
        except requests.RequestException as exc:
            raise RuntimeError(f"HTTP request to server failed: {exc}") from exc

        if resp.status_code != 200:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise RuntimeError(f"HTTP error from server: {resp.status_code} - {detail}")

        obs_dict = pickle.loads(resp.content)
        done = self.handle_done(obs_dict)
        obs = deserialize_data(obs_dict)
        if not isinstance(obs, dict):
            raise ValueError("Obs is not a dictionary")
        if self.save_result:
            self._record(obs, action_dict)
            self._record_episode_results(obs)
        step_time = time.time() - start
        self.step_count += 1
        self._update_web_frame(obs)

        # Progress indicator (update every 10 steps to reduce noise)
        if self.verbose and self.step_count % 10 == 0:
            time_color = (
                Colors.BRIGHT_GREEN
                if step_time < 0.1
                else Colors.YELLOW if step_time < 0.5 else Colors.RED
            )
            print(
                f"  {colored('⟳', Colors.DIM)} Step {colored(str(self.step_count), Colors.WHITE)}: "
                f"{colored(f'{step_time:.3f}s', time_color)}"
            )
        return obs, done

    def _update_web_frame(self, obs: dict) -> None:
        if not self._web_view:
            return
        if self._web_cv2 is None:
            return
        if self.step_count % self._web_view_interval != 0:
            return
        if not self.worker_ids:
            return
        wid = self.worker_ids[0]
        wdata = obs.get(wid, {})
        wobs = wdata.get("obs", {})
        if not wobs or wobs.get("reset"):
            return
        frames_by_cam = {}
        for k, v in list(wobs.items()):
            if isinstance(k, str) and k.startswith("video."):
                cam_name = k.split(".", 1)[1]
                if isinstance(v, np.ndarray):
                    frames_by_cam[cam_name] = v
        if not frames_by_cam:
            return
        try:
            top = concat_cams_top(frames_by_cam, self.cam_order)
        except Exception:
            return
        if self._web_view_scale != 1.0:
            h, w = top.shape[:2]
            new_w = max(2, int(w * self._web_view_scale))
            new_h = max(2, int(h * self._web_view_scale))
            top = self._web_cv2.resize(top, (new_w, new_h), interpolation=self._web_cv2.INTER_AREA)
        frame_bgr = self._web_cv2.cvtColor(top, self._web_cv2.COLOR_RGB2BGR)
        ok, buf = self._web_cv2.imencode(
            ".jpg", frame_bgr, [int(self._web_cv2.IMWRITE_JPEG_QUALITY), 80]
        )
        if not ok:
            return
        with self._web_frame_lock:
            self._web_frame_jpeg = buf.tobytes()
        if self._web_server is not None:
            self._web_server._frame_jpeg = self._web_frame_jpeg  # type: ignore[attr-defined]

    def handle_done(self, data: dict):
        for wobs in data.values():
            if wobs["obs"] is not None and wobs["obs"]["reset"]:
                self._print_eval_result(wobs["metric"], title="Episode Result")
        if all(
            [data[worker_id]["metric"] is not None for worker_id in self.worker_ids]
        ) and all(data[worker_id]["obs"] is None for worker_id in self.worker_ids):
            result_dict = {}
            for worker_data in data.values():
                for key, value in worker_data["metric"].items():
                    result_dict[key] = value
            # Filter out keys with "*"
            filtered_results = {k: v for k, v in result_dict.items() if "*" not in k}
            self._print_eval_result(filtered_results, title="Final Evaluation Result")
            return True
        return False

    def _print_eval_result(
        self, metrics: dict, title: str = "Evaluation Result"
    ) -> None:
        """Print formatted evaluation results."""
        if not metrics:
            return

        width = 48
        inner_width = width - 4

        # Header
        print(colored(f"{Box.TL}{Box.H * (width - 2)}{Box.TR}", Colors.MAGENTA))
        title_pad = (inner_width - len(title)) // 2
        title_line = (
            " " * title_pad
            + colored(title, Colors.BOLD, Colors.MAGENTA)
            + " " * (inner_width - title_pad - len(title))
        )
        print(make_box_line(title_line, inner_width, border_color=Colors.MAGENTA))
        print(colored(f"{Box.LT}{Box.H * (width - 2)}{Box.RT}", Colors.MAGENTA))

        # Metrics
        for key, value in metrics.items():

            # Format value
            if isinstance(value, float):
                if 0 <= value <= 1:
                    # Likely a rate/ratio - show as percentage with color
                    val_color = (
                        Colors.BRIGHT_GREEN
                        if value >= 0.8
                        else Colors.YELLOW if value >= 0.5 else Colors.RED
                    )
                    val_str = colored(f"{value:.2%}", val_color, Colors.BOLD)
                    val_len = 7
                else:
                    val_str = colored(f"{value:.4f}", Colors.WHITE)
                    val_len = len(f"{value:.4f}")
            else:
                val_str = colored(str(value), Colors.WHITE)
                val_len = len(str(value))

            key_display = key[:25] + ".." if len(key) > 25 else key
            key_len = min(len(key), 25) + (2 if len(key) > 25 else 0)
            padding = inner_width - 2 - key_len - 2 - val_len
            line_content = f" {key_display}" + " " * max(1, padding) + f"  {val_str}"
            print(
                make_box_line(line_content, inner_width, border_color=Colors.MAGENTA)
            )

        # Footer
        print(colored(f"{Box.BL}{Box.H * (width - 2)}{Box.BR}", Colors.MAGENTA))

    def kill_workers(self):
        resp = requests.post(
            f"{self.base_url}/kill",
            json={"data": {"worker_ids": self.worker_ids}},
            timeout=60,
            headers=self._build_headers(),
        )
        if resp.status_code != 200:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise RuntimeError(
                f"HTTP error on kill_workers: {resp.status_code} - {detail}"
            )


def fake_action(arm_type: str, gripper_type: str, control_type: str) -> dict:
    if arm_type == "franka":
        if gripper_type == "panda_hand":
            if control_type == "joint_position":
                actions = {
                    "action": [0.0] * 9,
                    "base_motion": [0.0, 0.0, 0.0],
                    "control_type": "joint_position",
                    "is_rel": True,
                }
            elif control_type == "ee_pose":
                actions = {
                    "action": (
                        [0.001, 0.001, 0.001],
                        [1.0, 0.0, 0.0, 0.0],
                        [0.04, 0.04],
                    ),
                    "base_motion": [0.0, 0.0, 0.0],
                    "control_type": "ee_pose",
                    "is_rel": True,
                }
            else:
                raise ValueError("Invalid control type")
        elif gripper_type == "robotiq":
            if control_type == "joint_position":
                actions = {
                    "action": [0] * 13,
                    "base_motion": [0.0, 0.0, 0.0],
                    "control_type": "joint_position",
                    "is_rel": True,
                }
            elif control_type == "ee_pose":
                actions = {
                    "action": (
                        [0.001, 0.001, 0.001],
                        [1.0, 0.0, 0.0, 0.0],
                        [0.7853, 0.7853, -0.7853, -0.7853, -0.7853, -0.7853],
                    ),
                    "base_motion": [0.0, 0.0, 0.0],
                    "control_type": "ee_pose",
                    "is_rel": True,
                }
            else:
                raise ValueError("Invalid control type")
        else:
            raise ValueError("Invalid gripper type")
    elif arm_type == "aloha":
        if gripper_type == "piper":
            if control_type == "joint_position":
                actions = {
                    "action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.05] * 2,
                    "base_motion": [0.0, 0.0, 0.0],
                    "control_type": "joint_position",
                    "is_rel": True,
                }
            elif control_type == "ee_pose":
                actions = {
                    "action": (
                        (
                            [0.001, 0.001, 0.001],
                            [1.0, 0.0, 0.0, 0.0],
                            [0.05, 0.05],
                        ),
                        (
                            [0.001, 0.001, 0.001],
                            [1.0, 0.0, 0.0, 0.0],
                            [0.05, 0.05],
                        ),
                    ),
                    "base_motion": [0.0, 0.0, 0.0],
                    "control_type": "ee_pose",
                    "is_rel": True,
                }
            else:
                raise ValueError("Invalid control type")
        else:
            raise ValueError("Invalid gripper type")
    elif arm_type == "r5a":
        if gripper_type == "lift2":
            if control_type == "joint_position":
                actions = {
                    "action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.04] * 2,
                    "base_motion": [0.0, 0.0, 0.0],
                    "control_type": "joint_position",
                    "is_rel": True,
                }
            elif control_type == "ee_pose":
                actions = {
                    "action": (
                        (
                            [0.001, 0.001, 0.001],
                            [1.0, 0.0, 0.0, 0.0],
                            [0.05, 0.05],
                        ),
                        (
                            [0.001, 0.001, 0.001],
                            [1.0, 0.0, 0.0, 0.0],
                            [0.05, 0.05],
                        ),
                    ),
                    "base_motion": [0.0, 0.0, 0.0],
                    "control_type": "ee_pose",
                    "is_rel": True,
                }
            else:
                raise ValueError("Invalid control type")
        else:
            raise ValueError("Invalid gripper type")
    else:
        raise ValueError("Invalid arm type")
    return actions


def _parse_list(s: str):
    return s.split(",")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worker_ids",
        type=_parse_list,
        default=["0"],
        help="List of worker IDs, i.e. --worker_ids 0,1,2",
    )
    parser.add_argument(
        "-cfg",
        "--config",
        type=lambda s: s.split(","),
        default=None,
        help="List of config paths, i.e. --config config1.yaml,config2.yaml",
    )
    parser.add_argument("--master", action="store_true")
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8087)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("-a", "--arm_type", type=str, default="franka")
    parser.add_argument("-g", "--gripper_type", type=str, default="panda_hand")
    parser.add_argument("-c", "--control_type", type=str, default="joint_position")
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="API token for authenticated eval servers",
    )
    parser.add_argument(
        "--robot_id",
        type=str,
        default=None,
        choices=list(ROBOT_ACTION_CONFIGS.keys()),
        help=f"Robot ID for action visualization, supported: {list(ROBOT_ACTION_CONFIGS.keys())}",
    )
    parser.add_argument(
        "--web-view",
        action="store_true",
        help="Start a lightweight web viewer for the stream",
    )
    parser.add_argument(
        "--web-view-port",
        type=int,
        default=8088,
        help="Web viewer port (default: 8088)",
    )
    parser.add_argument(
        "--web-view-interval",
        type=int,
        default=10,
        help="Show one frame every N steps (default: 10)",
    )
    parser.add_argument(
        "--web-view-scale",
        type=float,
        default=1.0,
        help="Scale factor for web viewer frames (default: 1.0)",
    )
    return parser


def run_cli(args: argparse.Namespace) -> int:
    if hasattr(args, "url") and args.url:
        base_url = args.url
    else:
        base_url = f"http://{args.host}:{args.port}"
    client = EvalClient(
        base_url,
        args.worker_ids,
        robot_id=args.robot_id,
        token=args.token,
        web_view=getattr(args, "web_view", False),
        web_view_port=getattr(args, "web_view_port", 8088),
        web_view_interval=getattr(args, "web_view_interval", 10),
        web_view_scale=getattr(args, "web_view_scale", 1.0),
    )

    try:
        _ = client.reset()
        while True:
            action = {
                i: fake_action(args.arm_type, args.gripper_type, args.control_type)
                for i in args.worker_ids
            }
            obs, done = client.step(action)
            if done or obs is None:
                break
            # Check if obs data is valid before accessing
            worker_obs = obs.get(args.worker_ids[0], {}).get("obs")
            if worker_obs is None:
                # No valid observation (e.g., server finished all tasks)
                break
            if worker_obs.get("reset"):
                pass
    finally:
        client.close()
    return 0
