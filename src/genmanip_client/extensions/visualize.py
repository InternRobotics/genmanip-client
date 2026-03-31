"""Local episode visualizer – self-contained web server using Rerun SDK.

Run with:
    gmp visualize [--project_root DIR] [--port 55077]

Routes
------
GET /                        – run list (home)
GET /run?id=RUN_DIR          – run detail (tasks / episodes card grid)
GET /episode?...             – episode detail + Rerun viewer iframe
GET /grpc-url?ep=EP_DIR      – start gRPC session, return {"url": "rerun+http://HOST:GRPC_PORT/proxy"}
GET /proxy?g=GRPC_PORT       – WebSocket proxy (wss→ws): browser wss:// ↔ 127.0.0.1:GRPC_PORT
GET /viewer/*                – cached Rerun WASM viewer assets
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pickle
import socket
import ssl
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, quote, unquote, urlparse
from uuid import uuid4

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

_CACHE_DIR   = Path.home() / ".cache" / "genmanip_vis"
_VIEWER_DIR  = _CACHE_DIR / "rerun_viewer"
_CERT_FILE   = _CACHE_DIR / "cert.pem"
_KEY_FILE    = _CACHE_DIR / "key.pem"


_SUCCESS_ICON = {True: "✓", False: "✗", None: "·"}

# ── Background gRPC session state ─────────────────────────────────────────────
# Rerun 0.30.x always binds port 9876 — we start the server ONCE and keep it
# alive for the lifetime of the visualizer.  Episode switching works by sending
# rr.Clear(recursive=True) then logging new data to the same RecordingStream.

_grpc_rec:  Optional[object] = None   # rr.RecordingStream (global, persistent)
_grpc_port: int              = 0
_grpc_path: str              = "/"
_grpc_lock      = threading.Lock()
_grpc_build_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="grpc_build")

# ── Per-episode .rrd file cache ───────────────────────────────────────────────
# Each episode is saved to an isolated .rrd file so that switching episodes
# never mixes timelines or static components from different episodes.

_RRD_CACHE_DIR = _CACHE_DIR / "episodes"
_rrd_lock   = threading.Lock()
_rrd_events: dict = {}   # ep_key → threading.Event (set when build done)
_rrd_ready:  dict = {}   # ep_key → Path | None
_rrd_hash_to_dir: dict = {}  # md5(ep_key) → Path


def _import_rerun():
    """Import rerun with a clearer error for unsupported Python versions."""
    try:
        import rerun as rr
        return rr
    except ImportError as e:
        msg = str(e)
        if "cannot import name 'Self' from 'typing'" in msg:
            raise RuntimeError(
                "rerun-sdk requires Python 3.11+ in this environment. "
                "Your current Python is too old for `gmp visualize`."
            ) from e
        raise RuntimeError(
            "rerun-sdk is not installed or failed to import. "
            'Install it with `pip install -e ".[visualize]"` in a compatible environment.'
        ) from e


# ── CSS / HTML helpers ─────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117;
       color: #e2e8f0; font-size: 14px; line-height: 1.6; }
a { color: #63b3ed; text-decoration: none; }
a:hover { text-decoration: underline; }
h1 { font-size: 1.4rem; font-weight: 700; margin-bottom: 0.5rem; }
h2 { font-size: 1.1rem; font-weight: 600; margin: 1.5rem 0 0.75rem; color: #90cdf4; }
.container { max-width: 1200px; margin: 0 auto; padding: 2rem 1.5rem; }
.header { border-bottom: 1px solid #2d3748; padding-bottom: 1rem; margin-bottom: 1.5rem; }
.header small { color: #718096; }
table { width: 100%; border-collapse: collapse; margin-top: 0.5rem; }
th { text-align: left; padding: 0.5rem 0.75rem; background: #1a202c;
     border-bottom: 2px solid #2d3748; font-weight: 600; color: #a0aec0; }
td { padding: 0.45rem 0.75rem; border-bottom: 1px solid #1e2533; }
tr:hover td { background: #1a2035; }
.badge-ok   { color: #68d391; font-weight: 600; }
.badge-fail { color: #fc8181; font-weight: 600; }
.badge-na   { color: #718096; }
.metric     { font-family: monospace; }
.tag        { display: inline-block; font-size: 0.75rem; background: #2d3748;
              border-radius: 4px; padding: 0 6px; margin-left: 4px; }
.card       { background: #1a202c; border: 1px solid #2d3748; border-radius: 8px;
              padding: 1.25rem 1.5rem; margin-bottom: 1rem; }
.breadcrumb { color: #718096; font-size: 0.85rem; margin-bottom: 1.5rem; }
.breadcrumb a { color: #a0aec0; }
.viewer-wrap { position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
               z-index: 1000; background: #0f1117; }
.viewer-wrap iframe { width: 100%; height: 100%; border: none; display: block; }
.viewer-back { position: fixed; top: 10px; left: 10px; z-index: 1001;
               background: rgba(26,32,44,0.9); color: #e2e8f0; border: 1px solid #2d3748;
               border-radius: 6px; padding: 6px 14px; font-size: 0.85rem;
               cursor: pointer; text-decoration: none; }
.viewer-back:hover { background: #2d3748; }
.metrics-grid { display: flex; flex-wrap: wrap; gap: 0.75rem; margin-bottom: 1rem; }
.metric-card { background: #1a202c; border: 1px solid #2d3748; border-radius: 6px;
               padding: 0.6rem 1rem; min-width: 140px; }
.metric-card .label { font-size: 0.75rem; color: #718096; }
.metric-card .value { font-size: 1.2rem; font-weight: 700; color: #90cdf4; }
.info-msg { color: #718096; padding: 2rem 0; text-align: center; }
"""

def _page(title: str, body: str) -> str:
    return (
        f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{title} – GenManip Visualizer</title>"
        f"<style>{_CSS}</style></head><body>"
        f"<div class='container'>{body}</div></body></html>"
    )

def _pct(v) -> str:
    if v is None:
        return "<span class='badge-na'>N/A</span>"
    try:
        f = float(v)
        cls = "badge-ok" if f >= 0.5 else "badge-fail"
        return f"<span class='{cls} metric'>{f*100:.0f}%</span>"
    except Exception:
        return str(v)


# ── Episode scanning ──────────────────────────────────────────────────────────

def _scan_run(run_dir: Path) -> dict:
    """Return {task_key: {"bench", "task", "episodes": [{id, dir, result_info}]}}."""
    _skip = {"__MACOSX"}
    tasks: dict[str, dict] = {}
    seen: set[Path] = set()

    for info_path in sorted(run_dir.rglob("result_info.json")):
        ep_dir    = info_path.parent
        task_dir  = ep_dir.parent
        bench_dir = task_dir.parent

        if bench_dir == run_dir or not bench_dir.is_relative_to(run_dir):
            continue
        if any(d.name in _skip or d.name.startswith(".") for d in (bench_dir, task_dir, ep_dir)):
            continue
        if ep_dir in seen:
            continue
        seen.add(ep_dir)

        result_info: dict = {}
        try:
            with open(info_path) as f:
                result_info = json.load(f)
        except Exception:
            pass

        task_key = f"{bench_dir.name} / {task_dir.name}"
        if task_key not in tasks:
            tasks[task_key] = {"bench": bench_dir.name, "task": task_dir.name, "episodes": []}
        tasks[task_key]["episodes"].append({
            "id": ep_dir.name,
            "dir": ep_dir,
            "result_info": result_info,
        })

    for t in tasks.values():
        t["episodes"].sort(key=lambda e: e["id"])
    return tasks


def _find_runs(base: Path) -> list[Path]:
    runs: list[Path] = []
    if not base.exists():
        return runs
    for bench in sorted(base.iterdir()):
        if not bench.is_dir() or bench.name.startswith("."):
            continue
        for r in sorted(bench.iterdir()):
            if r.is_dir() and not r.name.startswith("."):
                runs.append(r)
    return runs


# ── Rerun helpers ─────────────────────────────────────────────────────────────

def _ensure_cert(hostname: str = "") -> bool:
    """Generate a self-signed TLS certificate with SAN if not already present.

    Certs without a subjectAltName are rejected by modern browsers with no
    bypass option, causing all fetch() calls to the same HTTPS origin to fail.
    We auto-detect and regenerate old SAN-less certs on startup.
    """
    import ipaddress

    def _has_san(cert_path: Path) -> bool:
        try:
            r = subprocess.run(
                ["openssl", "x509", "-in", str(cert_path), "-noout", "-text"],
                capture_output=True, text=True,
            )
            return "Subject Alternative Name" in r.stdout
        except Exception:
            return True  # assume OK if openssl not available

    if _CERT_FILE.exists() and _KEY_FILE.exists():
        if _has_san(_CERT_FILE):
            return True
        print("  Regenerating TLS certificate (missing SAN)...")
        _CERT_FILE.unlink(missing_ok=True)
        _KEY_FILE.unlink(missing_ok=True)

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Build SAN string: always include localhost + 127.0.0.1; add server host too
    san_parts = ["DNS:localhost", "IP:127.0.0.1"]
    if hostname:
        try:
            ipaddress.ip_address(hostname)
            san_parts.append(f"IP:{hostname}")
        except ValueError:
            san_parts.append(f"DNS:{hostname}")
    san = ",".join(san_parts)

    try:
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", str(_KEY_FILE),
            "-out",    str(_CERT_FILE),
            "-days",   "3650",
            "-nodes",
            "-subj",   "/CN=genmanip-visualizer",
            "-addext", f"subjectAltName={san}",
        ], check=True, capture_output=True)
        print(f"  TLS certificate generated (SAN: {san})")
        return True
    except Exception as e:
        print(f"  Warning: could not generate TLS certificate: {e}")
        return False


def _cache_viewer_assets() -> bool:
    """Download rerun viewer assets (wasm/js/html) into _VIEWER_DIR. Returns True on success."""
    import time
    import urllib.request as urlreq

    _VIEWER_DIR.mkdir(parents=True, exist_ok=True)
    wasm_path    = _VIEWER_DIR / "re_viewer_bg.wasm"
    js_path      = _VIEWER_DIR / "re_viewer.js"
    html_path    = _VIEWER_DIR / "index.html"
    version_file = _VIEWER_DIR / ".sdk_version"

    try:
        rr = _import_rerun()
        current_version = rr.__version__
    except RuntimeError:
        current_version = "unknown"

    cached_version = version_file.read_text().strip() if version_file.exists() else ""

    wasm_ok    = wasm_path.exists() and wasm_path.stat().st_size > 20 * 1024 * 1024
    js_ok      = js_path.exists()   and js_path.stat().st_size  > 10 * 1024
    html_ok    = html_path.exists() and html_path.stat().st_size > 0
    version_ok = cached_version == current_version

    if wasm_ok and js_ok and html_ok and version_ok:
        return True

    if not version_ok and (wasm_ok or js_ok or html_ok):
        print(f"  Rerun SDK version changed ({cached_version!r} → {current_version!r}), refreshing viewer cache...")
        for stale in [wasm_path, js_path, html_path,
                      Path(str(wasm_path) + ".gz"), Path(str(js_path) + ".gz")]:
            stale.unlink(missing_ok=True)

    try:
        rr = _import_rerun()
        # Find a free port for the temporary asset server
        asset_port = 19090
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", asset_port)); break
                except OSError:
                    asset_port += 1

        print(f"  Starting Rerun asset server on port {asset_port}...")
        try:
            rr.start_web_viewer_server(port=asset_port)
        except AttributeError:
            rr.serve_web_viewer(web_port=asset_port, open_browser=False)
        time.sleep(2)

        base = f"http://127.0.0.1:{asset_port}"
        for filename, dest, min_size in [
            ("re_viewer_bg.wasm", wasm_path, 20 * 1024 * 1024),
            ("re_viewer.js",      js_path,   10 * 1024),
            ("index.html",        html_path, 0),
        ]:
            if dest.exists() and dest.stat().st_size >= min_size:
                pass
            else:
                print(f"  Downloading {filename}...")
                with urlreq.urlopen(f"{base}/{filename}", timeout=120) as resp:
                    data = resp.read()
                dest.write_bytes(data)
                print(f"  Cached {filename}: {len(data):,} bytes")
            # Pre-compress wasm and js for faster serving
            gz_dest = Path(str(dest) + ".gz")
            if not gz_dest.exists() and dest.suffix in (".wasm", ".js"):
                print(f"  Compressing {filename}...")
                with gzip.open(str(gz_dest), "wb", compresslevel=9) as f:
                    f.write(dest.read_bytes())
                print(f"  Compressed {filename}: {gz_dest.stat().st_size:,} bytes")
        # Patch index.html: expose WebHandle so the parent frame can call
        # add_receiver / remove_receiver without reloading the iframe.
        # (No fetch/WebSocket intercepts needed — the server is plain HTTP so
        #  gRPC-web fetch requests are never mixed-content.)
        try:
            content = html_path.read_text()
            patched = content.replace(
                "let handle = new wasm_bindgen.WebHandle(options);",
                "let handle = new wasm_bindgen.WebHandle(options); window._webhandle = handle;",
            )
            if patched != content:
                html_path.write_text(patched)
                print("  Patched index.html to expose _webhandle.")
        except Exception as pe:
            print(f"  Warning: could not patch index.html: {pe}")
        version_file.write_text(current_version)
        return True
    except Exception as e:
        print(f"  Warning: could not cache Rerun viewer assets: {e}")
        return False


def _load_pkl(episode_dir: Path) -> Optional[dict]:
    for pkl_file in sorted(episode_dir.glob("*.pkl")):
        if pkl_file.name.startswith("._"):
            continue
        try:
            with open(pkl_file, "rb") as f:
                return {"data": pickle.load(f), "filename": pkl_file.name}
        except Exception as e:
            return {"error": str(e), "filename": pkl_file.name}
    return None


def _category(name: str) -> str:
    """Map a data key name to one of: joint / gripper / base."""
    n = name.lower()
    if "gripper" in n:
        return "gripper"
    if "base" in n or "mobile" in n:
        return "base"
    return "joint"


def _log_curves_to(rec, data: dict, video_ts_ns) -> None:
    """Log timeseries curves from pkl data into a RecordingStream."""
    rr = _import_rerun()

    if not isinstance(data, dict):
        return
    n_steps = None
    for val in data.values():
        if isinstance(val, list) and val:
            n_steps = len(val)
            break
    if n_steps is None:
        return

    def make_time_col(T: int):
        if video_ts_ns is not None and len(video_ts_ns) >= T:
            return rr.TimeColumn("video_time", duration=1e-9 * video_ts_ns[:T])
        return rr.TimeColumn("step", sequence=list(range(T)))

    def _send(path, col, tc):
        rr.send_columns(path, indexes=[tc], columns=rr.Scalars.columns(scalars=col), recording=rec)

    # Look up robot config so we can split arm_action into arm vs gripper dimensions.
    # The pkl stores robot_id (e.g. "manip/franka/robotiq") and model_output[*]["arm_action"]
    # which is [arm_joints..., gripper_joints...] concatenated — without splitting, gripper
    # dims end up under joint/action/ instead of gripper/action/.
    robot_id = data.get("robot_id")
    robot_cfg = None
    if robot_id:
        try:
            from genmanip_client.vis_utils import ROBOT_ACTION_CONFIGS
            robot_cfg = ROBOT_ACTION_CONFIGS.get(robot_id)
            if robot_cfg:
                print(f"  [curves] robot_id={robot_id!r} cfg={robot_cfg}")
            else:
                print(f"  [curves] robot_id={robot_id!r} not in ROBOT_ACTION_CONFIGS, falling back")
        except ImportError:
            print("  [curves] Warning: could not import ROBOT_ACTION_CONFIGS from vis_utils")

    def _log_arm_action(arr: np.ndarray, tc) -> None:
        """Split arm_action array into arm / gripper (/ base) paths using robot_cfg."""
        if robot_cfg is None:
            # No config: log everything under joint/action as before
            for d in range(arr.shape[1]):
                _send(f"joint/action/arm_action/{d}", arr[:, d], tc)
            return
        cfg = robot_cfg
        ls, le = cfg.left_arm_slice
        gs, ge = cfg.left_gripper_slice
        prefix = "left_" if cfg.is_dual_arm else ""
        for d in range(le - ls):
            _send(f"joint/action/{prefix}arm/{d}", arr[:, ls + d], tc)
        for d in range(ge - gs):
            _send(f"gripper/action/{prefix}gripper/{d}", arr[:, gs + d], tc)
        if cfg.is_dual_arm and cfg.right_arm_slice and cfg.right_gripper_slice:
            rs, re = cfg.right_arm_slice
            rgs, rge = cfg.right_gripper_slice
            for d in range(re - rs):
                _send(f"joint/action/right_arm/{d}", arr[:, rs + d], tc)
            for d in range(rge - rgs):
                _send(f"gripper/action/right_gripper/{d}", arr[:, rgs + d], tc)

    for key, val in data.items():
        # Skip non-series fields
        if key in ("robot_id", "instruction") or not isinstance(val, list) or not val:
            continue
        first = val[0]
        T = len(val)
        cat = _category(key)
        try:
            if isinstance(first, np.ndarray) and first.ndim == 1:
                arr = np.stack(val).astype(np.float64)
                tc = make_time_col(T)
                for d in range(arr.shape[1]):
                    _send(f"{cat}/state/{key}/{d}", arr[:, d], tc)
            elif isinstance(first, dict):
                tc = make_time_col(T)
                for sk, sv0 in first.items():
                    scat = _category(sk)
                    if isinstance(sv0, np.ndarray) and sv0.ndim == 1:
                        try:
                            arr = np.stack([item[sk] for item in val]).astype(np.float64)
                        except Exception:
                            continue
                        if sk == "arm_action":
                            _log_arm_action(arr, tc)
                        else:
                            for d in range(arr.shape[1]):
                                _send(f"{scat}/action/{sk}/{d}", arr[:, d], tc)
                    elif isinstance(sv0, (list, tuple)) and sv0 and isinstance(sv0[0], (int, float, np.floating)):
                        # e.g. base_motion = [vx, vy, vyaw]
                        try:
                            arr = np.array([item[sk] for item in val], dtype=np.float64)
                            for d in range(arr.shape[1]):
                                _send(f"{scat}/action/{sk}/{d}", arr[:, d], tc)
                        except Exception:
                            pass
                    elif isinstance(sv0, (int, float)) and not isinstance(sv0, bool):
                        arr = np.array([item[sk] for item in val], dtype=np.float64)
                        _send(f"{scat}/action/{sk}", arr, tc)
            elif isinstance(first, (int, float)) and not isinstance(first, bool):
                arr = np.array(val, dtype=np.float64)
                rr.send_columns(
                    f"{cat}/state/{key}", indexes=[make_time_col(T)],
                    columns=rr.Scalars.columns(scalars=arr),
                    recording=rec,
                )
        except Exception as e:
            print(f"  Warning: error logging {key}: {e}")


def _ensure_grpc_server() -> tuple[int, str]:
    """Start the global gRPC server if not yet running. Returns (grpc_port, grpc_path)."""
    global _grpc_rec, _grpc_port, _grpc_path
    with _grpc_lock:
        if _grpc_rec is not None:
            return _grpc_port, _grpc_path
        try:
            rr = _import_rerun()
        except RuntimeError as e:
            raise RuntimeError(str(e)) from e
        rec = rr.RecordingStream(
            application_id="genmanip-visualizer",
            recording_id="persistent",
            make_default=False, make_thread_default=False,
        )
        try:
            grpc_url = rr.serve_grpc(recording=rec, open_browser=False)
        except TypeError:
            grpc_url = rr.serve_grpc(recording=rec)
        print(f"  [serve_grpc] started: {grpc_url!r}")
        parsed = urlparse(grpc_url.replace("rerun+", ""))
        _grpc_port = parsed.port or 9876
        _grpc_path = parsed.path or "/"
        _grpc_rec = rec
        return _grpc_port, _grpc_path


def _start_grpc_session(episode_dir: Path) -> tuple[int, str]:
    """Ensure the global gRPC server is up, then (re-)log episode_dir.

    Rerun 0.30.x always binds port 9876 and cannot be restarted in the same
    process.  We keep ONE RecordingStream alive forever and, for each episode
    switch, send rr.Clear(recursive=True) followed by the new episode data.
    The viewer stays running; data arrives via the existing WebSocket connection.

    Returns (grpc_port, grpc_path).
    """
    grpc_port, grpc_path = _ensure_grpc_server()
    # Serialise logging through a single-worker pool so Clear always arrives
    # before the new data, even when the user clicks quickly.
    _grpc_build_pool.submit(_clear_and_log_episode, episode_dir)
    return grpc_port, grpc_path


def _clear_and_log_episode(episode_dir: Path) -> None:
    """No-op stub — episode switching now uses per-episode .rrd files."""
    pass


def _log_episode_to_grpc(episode_dir: Path, rec) -> None:
    """Log all episode data into an existing gRPC-connected RecordingStream."""
    try:
        rr = _import_rerun()
        import rerun.blueprint as rrb
    except RuntimeError:
        return

    video_files = sorted(v.name for v in episode_dir.glob("*.mp4") if not v.name.startswith("._"))
    pkl_data    = _load_pkl(episode_dir)
    data        = (pkl_data or {}).get("data") or {}

    if pkl_data and pkl_data.get("error"):
        print(f"  Warning: could not load pkl: {pkl_data['error']}")

    cam_views = [
        rrb.Spatial2DView(name=v.replace(".mp4", ""), origin=f"camera/{v.replace('.mp4','')}")
        for v in video_files
    ]
    state_row  = rrb.Horizontal(
        rrb.TimeSeriesView(name="Joint State",   origin="joint/state"),
        rrb.TimeSeriesView(name="Gripper State", origin="gripper/state"),
        rrb.TimeSeriesView(name="Base State",    origin="base/state"),
    )
    action_row = rrb.Horizontal(
        rrb.TimeSeriesView(name="Joint Action",   origin="joint/action"),
        rrb.TimeSeriesView(name="Gripper Action", origin="gripper/action"),
        rrb.TimeSeriesView(name="Base Action",    origin="base/action"),
    )
    curve_rows = rrb.Vertical(state_row, action_row)
    if cam_views and data:
        blueprint = rrb.Blueprint(
            rrb.Vertical(rrb.Horizontal(*cam_views), curve_rows, row_shares=[3, 2]),
            auto_layout=False, auto_views=False,
        )
    elif cam_views:
        blueprint = rrb.Blueprint(rrb.Horizontal(*cam_views), auto_layout=False, auto_views=False)
    else:
        blueprint = rrb.Blueprint(curve_rows, auto_layout=False, auto_views=False)

    rr.send_blueprint(blueprint, make_active=True, make_default=True, recording=rec)

    # Pre-load all video assets in parallel (pure I/O)
    def _load_video(vname):
        path = episode_dir / vname
        if not path.exists():
            return None
        try:
            asset = rr.AssetVideo(path=str(path))
            return vname, asset, asset.read_frame_timestamps_nanos()
        except Exception as e:
            print(f"  Warning: could not load {vname}: {e}")
            return None

    with ThreadPoolExecutor() as ex:
        loaded = [r for r in ex.map(_load_video, video_files) if r is not None]

    video_ts_ns = loaded[0][2] if loaded else None

    for vname, asset, ts_ns in loaded:
        cam = vname.replace(".mp4", "")
        try:
            rr.send_columns(
                f"camera/{cam}",
                indexes=[rr.TimeColumn("video_time", duration=1e-9 * ts_ns)],
                columns=rr.VideoFrameReference.columns_nanos(ts_ns),
                recording=rec,
            )
            rr.log(f"camera/{cam}", asset, static=True, recording=rec)
        except Exception as e:
            print(f"  Warning: could not log {vname}: {e}")

    _log_curves_to(rec, data, video_ts_ns)
    rec.flush()


# ── Per-episode .rrd builder ──────────────────────────────────────────────────

def _build_rrd_for_ep(ep_dir: Path) -> Optional[Path]:
    """Build and cache a .rrd file for ep_dir.  Returns path or None on error."""
    from uuid import uuid4
    try:
        rr = _import_rerun()
        import rerun.blueprint as rrb
    except RuntimeError:
        return None

    # Store the .rrd next to the episode data so it lives on the same
    # large-storage mount as the mp4/pkl files, not in ~/.cache.
    rrd_path = ep_dir / ".genmanip_vis.rrd"

    if rrd_path.exists() and rrd_path.stat().st_size > 0:
        return rrd_path

    video_files = sorted(v.name for v in ep_dir.glob("*.mp4") if not v.name.startswith("._"))
    pkl_data    = _load_pkl(ep_dir)
    data        = (pkl_data or {}).get("data") or {}

    cam_views = [
        rrb.Spatial2DView(name=v.replace(".mp4", ""), origin=f"camera/{v.replace('.mp4','')}")
        for v in video_files
    ]
    state_row  = rrb.Horizontal(
        rrb.TimeSeriesView(name="Joint State",   origin="joint/state"),
        rrb.TimeSeriesView(name="Gripper State", origin="gripper/state"),
        rrb.TimeSeriesView(name="Base State",    origin="base/state"),
    )
    action_row = rrb.Horizontal(
        rrb.TimeSeriesView(name="Joint Action",   origin="joint/action"),
        rrb.TimeSeriesView(name="Gripper Action", origin="gripper/action"),
        rrb.TimeSeriesView(name="Base Action",    origin="base/action"),
    )
    curve_rows = rrb.Vertical(state_row, action_row)
    if cam_views and data:
        blueprint = rrb.Blueprint(
            rrb.Vertical(rrb.Horizontal(*cam_views), curve_rows, row_shares=[3, 2]),
            auto_layout=False, auto_views=False,
        )
    elif cam_views:
        blueprint = rrb.Blueprint(rrb.Horizontal(*cam_views), auto_layout=False, auto_views=False)
    else:
        blueprint = rrb.Blueprint(curve_rows, auto_layout=False, auto_views=False)

    rec = rr.RecordingStream(
        application_id="genmanip-visualizer",
        recording_id=uuid4().hex,
        make_default=False, make_thread_default=False,
    )
    try:
        rr.save(str(rrd_path), default_blueprint=blueprint, recording=rec)
    except TypeError:
        rr.save(str(rrd_path), recording=rec)

    def _load_video(vname):
        vpath = ep_dir / vname
        if not vpath.exists():
            return None
        try:
            asset = rr.AssetVideo(path=str(vpath))
            return vname, asset, asset.read_frame_timestamps_nanos()
        except Exception as e:
            print(f"  Warning: could not load {vname}: {e}")
            return None

    with ThreadPoolExecutor() as ex:
        loaded = [r for r in ex.map(_load_video, video_files) if r is not None]

    video_ts_ns = loaded[0][2] if loaded else None

    for vname, asset, ts_ns in loaded:
        cam = vname.replace(".mp4", "")
        try:
            rr.send_columns(
                f"camera/{cam}",
                indexes=[rr.TimeColumn("video_time", duration=1e-9 * ts_ns)],
                columns=rr.VideoFrameReference.columns_nanos(ts_ns),
                recording=rec,
            )
            rr.log(f"camera/{cam}", asset, static=True, recording=rec)
        except Exception as e:
            print(f"  Warning: could not log {vname}: {e}")

    _log_curves_to(rec, data, video_ts_ns)
    rec.flush()
    print(f"  [rrd] done: {rrd_path.name} ({rrd_path.stat().st_size//1024} KB)")
    return rrd_path


def _ep_hash(ep_dir: Path) -> str:
    return hashlib.md5(str(ep_dir).encode()).hexdigest()


def _start_rrd_build(ep_dir: Path) -> None:
    """Submit rrd build job if not already started or cached."""
    key  = str(ep_dir)
    h    = _ep_hash(ep_dir)
    rrd_path = ep_dir / ".genmanip_vis.rrd"
    with _rrd_lock:
        _rrd_hash_to_dir[h] = ep_dir
    if rrd_path.exists() and rrd_path.stat().st_size > 0:
        # Already on disk — mark as ready immediately
        with _rrd_lock:
            if key not in _rrd_events:
                ev = threading.Event()
                ev.set()
                _rrd_events[key] = ev
                _rrd_ready[key]  = rrd_path
        return
    with _rrd_lock:
        if key in _rrd_events:
            return
        ev = threading.Event()
        _rrd_events[key] = ev

    def _worker():
        try:
            result = _build_rrd_for_ep(ep_dir)
        except Exception as exc:
            import traceback
            print(f"  [rrd] ERROR building {ep_dir}: {exc}")
            traceback.print_exc()
            result = None
        with _rrd_lock:
            _rrd_ready[key] = result
        ev.set()
    _grpc_build_pool.submit(_worker)


def _await_rrd(ep_dir: Path) -> Optional[Path]:
    """Wait for the rrd build to complete; return path or None."""
    _start_rrd_build(ep_dir)
    key = str(ep_dir)
    with _rrd_lock:
        ev = _rrd_events.get(key)
    if ev:
        ev.wait()
    return _rrd_ready.get(key)


# ── HTTP handler ──────────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    """Routes: / /run /episode /grpc-url /viewer/*"""

    protocol_version = "HTTP/1.1"   # required for Transfer-Encoding: chunked
    server_state: dict  # injected by run_visualizer

    def log_message(self, fmt, *args):  # suppress default access log
        pass

    # ---------- routing ----------

    def do_POST(self):
        """Forward gRPC-web POST requests to the active session's gRPC server.

        The JS fetch intercept rewrites Rerun's gRPC fetch calls from
        http://HOST:GRPC_PORT/ServiceName/Method  →  https://HOST:MAIN_PORT/relay
        to avoid Chrome's h2 ALPN forcing for gRPC service URL patterns.
        The original gRPC path is passed in the x-grpc-path header.
        """
        active_port = self.server_state.get("_active_grpc_port", 0)
        path = urlparse(self.path).path
        if path == "/relay":
            path = self.headers.get("x-grpc-path") or "/"
        print(f"  [POST] raw_path={self.path!r} grpc_path={path!r} active_port={active_port} ct={self.headers.get('content-type')!r} cl={self.headers.get('content-length')!r}")
        # Drop Rerun catalog/telemetry calls — these are not data-stream calls and
        # forwarding them to the gRPC server causes "Failed to load entries" in the
        # viewer's sources panel by adding 127.0.0.1 as a (non-functional) catalog.
        _CATALOG_PREFIXES = ("/rerun.cloud.", "/api/")
        if any(path.startswith(p) for p in _CATALOG_PREFIXES):
            try:
                cl = int(self.headers.get("Content-Length", 0) or 0)
                if cl > 0:
                    self.rfile.read(cl)
            except Exception:
                pass
            self.send_response(503)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if not active_port:
            # No episode selected yet; drain body and return empty success.
            try:
                cl = int(self.headers.get("Content-Length", 0) or 0)
                if cl > 0:
                    self.rfile.read(cl)
            except Exception:
                pass
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self._serve_grpc_proxy(active_port, path)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Expose-Headers", "*")
        self.send_header("Access-Control-Max-Age", "86400")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        qs     = parse_qs(parsed.query)
        path   = parsed.path.rstrip("/") or "/"

        try:
            if path == "/":
                self._home(qs)
            elif path == "/run":
                self._run_detail(qs)
            elif path == "/episode":
                self._episode_detail(qs)
            elif path == "/grpc-url":
                self._serve_grpc_url(qs)
            elif path.startswith("/rrd/"):
                self._serve_rrd_file(path[5:].removesuffix('.rrd'))
            elif path == "/relay":
                # GET relay calls are Rerun startup health/version probes.
                # Return 404 so Rerun does NOT register 127.0.0.1 as a catalog source.
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
            elif path == "/ws-relay":
                self._serve_ws_relay(qs)
            elif path.startswith("/viewer"):
                self._serve_viewer(path[len("/viewer"):], parsed.query)
            else:
                self._send_404()
        except Exception as e:
            self._send_error(500, str(e))

    # ---------- pages ----------

    def _home(self, qs):
        import datetime
        runs = list(self.server_state["runs"])
        sort_by  = (qs.get("sort")  or ["time"])[0]   # "name" or "time"
        sort_dir = (qs.get("order") or ["desc"])[0]   # "asc" or "desc"
        reverse  = sort_dir == "desc"

        if sort_by == "name":
            runs.sort(key=lambda r: r.name, reverse=reverse)
        else:
            runs.sort(key=lambda r: r.stat().st_mtime, reverse=reverse)

        def _th(label, col):
            """Column header with sort toggle link."""
            if sort_by == col:
                new_order = "asc" if sort_dir == "desc" else "desc"
                arrow = " ▼" if sort_dir == "desc" else " ▲"
            else:
                new_order = "desc"
                arrow = ""
            return f"<th><a href='/?sort={col}&order={new_order}' style='color:inherit'>{label}{arrow}</a></th>"

        rows = ""
        for i, r in enumerate(runs):
            href  = f"/run?id={r}"
            mtime = datetime.datetime.fromtimestamp(r.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            rows += (
                f"<tr><td>{i+1}</td>"
                f"<td><a href='{href}'>{r.parent.name}</a></td>"
                f"<td><a href='{href}'>{r.name}</a></td>"
                f"<td>{mtime}</td></tr>\n"
            )
        body = (
            "<div class='header'><h1>GenManip Visualizer</h1>"
            "<small>Select a run to browse episodes</small></div>"
            f"<table><thead><tr><th>#</th>{_th('Benchmark','bench')}{_th('Run ID','name')}{_th('Created','time')}</tr></thead>"
            f"<tbody>{rows or '<tr><td colspan=4 class=info-msg>No runs found.</td></tr>'}</tbody></table>"
        )
        self._html(200, _page("Home", body))

    def _run_detail(self, qs):
        run_id = (qs.get("id") or [None])[0]
        if not run_id:
            self._send_404(); return

        run_dir = Path(run_id)
        if not run_dir.is_dir():
            self._send_404(); return

        tasks = _scan_run(run_dir)
        if not tasks:
            body = (
                f"<div class='breadcrumb'><a href='/'>Home</a> › {run_dir.name}</div>"
                "<p class='info-msg'>No episodes found in this run.</p>"
            )
            self._html(200, _page(run_dir.name, body)); return

        port   = self.server_state["port"]
        scheme = self.server_state["scheme"]
        host   = self.headers.get("Host", f"localhost:{port}")
        has_viewer = (_VIEWER_DIR / "index.html").exists()

        # Group tasks by benchmark
        benches: dict[str, list] = {}
        for task_key in sorted(tasks.keys()):
            tinfo = tasks[task_key]
            benches.setdefault(tinfo["bench"], []).append((task_key, tinfo))

        total_ep = total_ok = 0
        sidebar_html = ""

        for bench_name, bench_tasks in benches.items():
            bench_ep = bench_ok = 0
            bench_cards = ""
            for _, tinfo in bench_tasks:
                episodes = tinfo["episodes"]
                n  = len(episodes)
                ok = sum(1 for e in episodes if e["result_info"].get("success_rate", 0) >= 0.5)
                bench_ep += n; bench_ok += ok

                sr_pct = f"{ok/n*100:.0f}%" if n else "N/A"
                badge_color = (
                    "#718096" if n == 0 else
                    "#38a169" if ok == n else
                    "#d69e2e" if ok > 0 else
                    "#e53e3e"
                )

                dots = ""
                for ep in episodes:
                    ep_dir = ep["dir"]
                    metrics = {k: v for k, v in ep["result_info"].items() if k != "log_info"}
                    metrics_attr = json.dumps(metrics).replace("&", "&amp;").replace('"', "&quot;")
                    ep_dir_attr  = str(ep_dir).replace("&", "&amp;").replace('"', "&quot;")
                    ep_id   = ep["id"]
                    success = ep["result_info"].get("success_rate", 0) >= 0.5
                    color   = "#2ea44f" if success else "#e53e3e"
                    dots += (
                        f"<span class='ep-dot' tabindex='0' title='Episode {ep_id}' "
                        f"data-ep-dir=\"{ep_dir_attr}\" data-ep=\"{ep_id}\" data-metrics=\"{metrics_attr}\" "
                        f"style='display:inline-block;width:14px;height:14px;cursor:pointer;"
                        f"background:{color};border-radius:2px;margin:1px'></span>"
                    )

                task_short = tinfo["task"]
                bench_cards += (
                    f"<div style='background:#1a202c;border:1px solid #2d3748;border-radius:8px;"
                    f"overflow:hidden;margin-bottom:0.6rem'>"
                    f"<div style='padding:0.5rem 0.75rem;border-bottom:1px solid #2d3748;"
                    f"display:flex;justify-content:space-between;align-items:center;gap:0.5rem'>"
                    f"<span style='font-size:0.8rem;font-weight:600;overflow:hidden;text-overflow:ellipsis;"
                    f"white-space:nowrap;flex:1' title='{task_short}'>{task_short}</span>"
                    f"<span style='white-space:nowrap;font-size:0.7rem;color:#a0aec0'>{n} ep</span>"
                    f"<span style='background:{badge_color};color:#fff;font-size:0.7rem;"
                    f"font-weight:700;padding:1px 6px;border-radius:4px'>{sr_pct}</span>"
                    f"</div>"
                    f"<div style='padding:0.5rem 0.75rem;line-height:1'>{dots}</div>"
                    f"</div>"
                )

            total_ep += bench_ep; total_ok += bench_ok
            bench_sr = f"{bench_ok/bench_ep*100:.2f}%" if bench_ep else "N/A"
            sidebar_html += (
                f"<div style='margin-bottom:1rem'>"
                f"<div style='font-size:0.8rem;font-weight:600;color:#90cdf4;"
                f"display:flex;justify-content:space-between;margin-bottom:0.4rem'>"
                f"<span>{bench_name}</span>"
                f"<span style='color:#718096;font-weight:400'>{bench_sr}</span></div>"
                f"{bench_cards}</div>"
            )

        overall_sr = f"{total_ok/total_ep*100:.2f}%" if total_ep else "N/A"
        viewer_src = f"{scheme}://{host}/viewer/index.html"
        viewer_iframe = (
            f"<iframe id='rr-viewer' allow='cross-origin-isolated' src='{viewer_src}'"
            f" style='flex:1;width:100%;border:none;display:block'></iframe>"
            if has_viewer else
            "<div style='flex:1;display:flex;align-items:center;justify-content:center;"
            "color:#fc8181'>Rerun viewer assets not cached.</div>"
        )

        html = (
            f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{run_dir.name} – GenManip Visualizer</title>"
            f"<style>{_CSS}"
            f".split{{display:flex;height:100vh;overflow:hidden}}"
            f".sidebar{{width:300px;flex-shrink:0;display:flex;flex-direction:column;"
            f"border-right:1px solid #2d3748;overflow:hidden}}"
            f".sidebar-hdr{{padding:0.75rem 1rem;border-bottom:1px solid #2d3748;flex-shrink:0}}"
            f".sidebar-body{{flex:1;overflow-y:auto;padding:0.75rem 1rem}}"
            f".main{{flex:1;min-width:0;display:flex;flex-direction:column}}"
            f".infobar{{padding:0.35rem 1rem;border-bottom:1px solid #2d3748;font-size:0.8rem;"
            f"color:#a0aec0;flex-shrink:0;min-height:2rem;display:flex;align-items:center;gap:0.5rem}}"
            f".viewer-area{{flex:1;display:flex;flex-direction:column}}"
            f".ep-dot:hover{{opacity:0.75}}"
            f".ep-dot.active{{outline:2px solid #fff;outline-offset:1px}}"
            f"</style></head><body>"
            f"<div class='split'>"
            f"<div class='sidebar'>"
            f"<div class='sidebar-hdr'>"
            f"<div><a href='/'>← Home</a></div>"
            f"<div style='font-weight:700;margin-top:0.2rem;font-size:0.95rem'>{run_dir.name}</div>"
            f"<div style='font-size:0.72rem;color:#718096'>{total_ep} episodes · SR {overall_sr}</div>"
            f"</div>"
            f"<div class='sidebar-body'>{sidebar_html}</div>"
            f"</div>"
            f"<div class='main'>"
            f"<div id='infobar' class='infobar'>Opening viewer… click an episode dot to load a recording</div>"
            f"<div class='viewer-area'>{viewer_iframe}</div>"
            f"</div></div>"
            f"<script>"
            f"var activeDot=null;"
            f"window.addEventListener('message',function(e){{"
            f"  if(e.data!=='READY')return;"
            f"  if(!activeDot)document.getElementById('infobar').textContent='Click an episode dot to load a recording';"
            f"}});"
            f"function fetchAndLoad(epDir,epId,metrics){{"
            f"  var bar=document.getElementById('infobar');"
            f"  bar.textContent='Building recording…';"
            f"  fetch('/grpc-url?ep='+encodeURIComponent(epDir))"
            f"    .then(function(r){{if(!r.ok)throw new Error('HTTP '+r.status);return r.json();}}).then(function(d){{"
            f"      if(d.error){{bar.textContent='Error: '+d.error;console.error('grpc-url error',d.error);return;}}"
            f"      console.log('rrd-url',d.url);"
            f"      var tags=Object.entries(metrics).map(function(e){{"
            f"        var v=e[1];"
            f"        if(typeof v==='number'&&v>=0&&v<=1)v=Math.round(v*100)+'%';"
            f"        return '<span class=tag>'+e[0]+': '+v+'</span>';"
            f"      }}).join('');"
            f"      bar.innerHTML='<strong>'+epId+'</strong> <span style=color:#718096;font-size:0.75rem>loading…</span>'+tags;"
            f"      var frame=document.getElementById('rr-viewer');"
            f"      frame.onload=function(){{bar.innerHTML='<strong>'+epId+'</strong>'+tags;}};"
            f"      frame.src='/viewer/index.html?url='+encodeURIComponent(d.url)+'&t='+Date.now();"
            f"    }}).catch(function(e){{console.error('grpc-url fetch failed',e);bar.textContent='Error: '+e.message;}});"
            f"}}"
            f"function loadEp(epDir,epId,metrics){{"
            f"  fetchAndLoad(epDir,epId,metrics);"
            f"}}"
            f"document.addEventListener('click',function(e){{"
            f"  var d=e.target.closest('.ep-dot');"
            f"  if(!d)return;"
            f"  if(activeDot)activeDot.classList.remove('active');"
            f"  d.classList.add('active');activeDot=d;"
            f"  loadEp(d.dataset.epDir,d.dataset.ep,JSON.parse(d.dataset.metrics));"
            f"}});"
            f"</script></body></html>"
        )
        self._html(200, html, coep=True)

    def _episode_detail(self, qs):
        run_id  = (qs.get("run")    or [None])[0]
        bench   = (qs.get("bench")  or [None])[0]
        task    = (qs.get("task")   or [None])[0]
        ep_dir_s = (qs.get("ep_dir") or [None])[0]

        if not ep_dir_s:
            self._send_404(); return

        ep_dir = Path(ep_dir_s)
        if not ep_dir.is_dir():
            self._send_error(404, f"Episode directory not found: {ep_dir_s}"); return

        # Load result_info
        result_info: dict = {}
        try:
            with open(ep_dir / "result_info.json") as f:
                result_info = json.load(f)
        except Exception:
            pass

        # Metric cards
        metric_cards = ""
        for k, v in result_info.items():
            if k == "log_info":
                continue
            if isinstance(v, float) and 0 <= v <= 1:
                disp = f"{v*100:.0f}%"
            else:
                disp = str(v)
            metric_cards += (
                f"<div class='metric-card'>"
                f"<div class='label'>{k}</div>"
                f"<div class='value'>{disp}</div></div>"
            )

        # Rerun viewer iframe
        port   = self.server_state["port"]
        scheme = self.server_state["scheme"]
        host   = self.headers.get("Host", f"localhost:{port}")
        has_viewer = (_VIEWER_DIR / "index.html").exists()

        ep_dir_js = json.dumps(str(ep_dir))  # safely quoted for embedding in JS

        if has_viewer:
            viewer_src = f"{scheme}://{host}/viewer/index.html"
            run_id_enc2 = (run_id or "").replace("&", "&amp;")
            back2 = f"/run?id={run_id_enc2}" if run_id_enc2 else "/"
            viewer_block = (
                f"<div class='viewer-wrap'>"
                f"<a class='viewer-back' href='{back2}'>← Back</a>"
                f"<iframe id='rr-viewer' src='{viewer_src}' allow='cross-origin-isolated'></iframe>"
                f"</div>"
                f"<script>"
                f"var _rrdUrl=null;"
                f"window.addEventListener('message',function(e){{"
                f"  if(e.data!=='READY')return;"
                f"  fetch('/grpc-url?ep='+encodeURIComponent({ep_dir_js}))"
                f"    .then(function(r){{return r.json();}}).then(function(d){{"
                f"      var h=document.getElementById('rr-viewer').contentWindow;"
                f"      if(h&&h._webhandle){{"
                f"        if(_rrdUrl)h._webhandle.remove_receiver(_rrdUrl);"
                f"        h._webhandle.add_receiver(d.url,false);"
                f"        _rrdUrl=d.url;"
                f"      }}"
                f"    }}).catch(function(e){{console.error('rrd-url',e);}});"
                f"}});"
                f"</script>"
            )
        else:
            viewer_block = (
                f"<div class='card'><p style='color:#fc8181'>Rerun viewer assets not cached yet.</p>"
                f"<p style='margin-top:0.5rem'>Start the visualizer again to download viewer assets.</p></div>"
            )

        run_id_enc = (run_id or "").replace("&", "&amp;")
        back = f"/run?id={run_id_enc}" if run_id_enc else "/"
        ep_id = ep_dir.name
        body = (
            f"<div class='breadcrumb'><a href='/'>Home</a> › "
            f"<a href='{back}'>{bench or ''} / {task or ''}</a> › {ep_id}</div>"
            f"<div class='header'><h1>{ep_id}</h1><small>{ep_dir}</small></div>"
            f"<div class='metrics-grid'>{metric_cards}</div>"
            f"{viewer_block}"
        )
        self._html(200, _page(ep_id, body), coep=True)

    # ---------- data endpoints ----------

    def _serve_grpc_url(self, qs):
        """Start building the per-episode .rrd file; return its URL immediately.

        The browser reloads the viewer iframe with ?url=<rrd_url> which fetches
        /rrd/<hash> (blocking until the build finishes).  The WASM is
        browser-cached after the first load so subsequent switches are fast.
        """
        ep_dir_s = (qs.get("ep") or [None])[0]
        if not ep_dir_s:
            self._send_json_error(400, "Missing ep parameter"); return

        ep_dir = Path(ep_dir_s)
        if not ep_dir.is_dir():
            self._send_json_error(404, f"Episode directory not found: {ep_dir_s}"); return

        _start_rrd_build(ep_dir)   # kick off build (no-op if already started/cached)

        scheme = self.server_state["scheme"]
        host   = self.headers.get("Host", f"localhost:{self.server_state['port']}")
        rrd_url = f"{scheme}://{host}/rrd/{_ep_hash(ep_dir)}.rrd"

        data = json.dumps({"url": rrd_url}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _send_json_error(self, code: int, msg: str):
        body = json.dumps({"error": msg}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_rrd_file(self, ep_hash: str):
        """Serve the .rrd bytes for the episode identified by hash."""
        with _rrd_lock:
            ep_dir = _rrd_hash_to_dir.get(ep_hash)
        if ep_dir is None:
            self._send_error(404, "Unknown episode hash"); return
        rrd_path = _await_rrd(ep_dir)
        if not rrd_path or not rrd_path.exists():
            self._send_error(500, "RRD build failed"); return
        size = rrd_path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        self.end_headers()
        try:
            with open(rrd_path, "rb") as fh:
                while True:
                    chunk = fh.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        except (ssl.SSLError, BrokenPipeError, ConnectionResetError):
            pass  # client disconnected mid-transfer — normal for large .rrd files

    def _serve_grpc_proxy(self, grpc_port: int, grpc_path: str):
        """HTTP reverse proxy for gRPC-web fetch requests.

        The fetch monkey-patch in index.html rewrites
            fetch("http://HOST:GRPC_PORT/service/method")
        to
            fetch("https://HOST:MAIN_PORT/grpc-proxy/GRPC_PORT/service/method")
        and we forward it to http://127.0.0.1:GRPC_PORT/service/method.

        gRPC-web uses HTTP POST with streaming responses; we read chunks and
        forward them as they arrive so the viewer gets data incrementally.
        """
        import http.client

        # Read request body (gRPC-web POST body contains the serialised request)
        try:
            content_length = int(self.headers.get("Content-Length", 0) or 0)
        except ValueError:
            content_length = 0
        body = self.rfile.read(content_length) if content_length > 0 else b""

        # Forward headers (drop hop-by-hop headers)
        _HOP = {"host", "connection", "transfer-encoding", "te", "trailers",
                 "keep-alive", "proxy-authorization", "proxy-authenticate", "upgrade",
                 "content-length"} # Drop content length to avoid issues if we change body
        fwd_headers = {k: v for k, v in self.headers.items() if k.lower() not in _HOP}
        if content_length > 0:
             fwd_headers["Content-Length"] = str(content_length)
        # Rerun gRPC requires the origin to not be mismatched
        fwd_headers["Origin"] = f"http://127.0.0.1:{grpc_port}"
        # Rerun's Rust gRPC server specifically checks Host headers against the listener
        fwd_headers["Host"] = f"127.0.0.1:{grpc_port}"

        try:
            conn = http.client.HTTPConnection("127.0.0.1", grpc_port, timeout=300)
            conn.request(self.command, grpc_path, body, fwd_headers)
            resp = conn.getresponse()
            print(f"  [GRPC-PROXY] :{grpc_port}{grpc_path} → {resp.status} headers={dict(resp.getheaders())}")

            self.send_response(resp.status)
            for hdr, val in resp.getheaders():
                if hdr.lower() not in ("connection", "transfer-encoding", "keep-alive"):
                    self.send_header(hdr, val)
            
            # Since the browser makes requests to the proxy (port 55077) from the main page (port 55077),
            # this is NOT a cross-origin request anymore.
            # But the iframe inside might have origin issues.
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Expose-Headers", "*")
            
            # gRPC-web streams require chunked transfer encoding or no content-length
            # If the upstream didn't provide a content-length, we MUST NOT send one.
            # We also ensure connection is closed.
            self.send_header("Connection", "close")
            self.end_headers()

            try:
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()  # flush every chunk: gRPC frames are small
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                conn.close()
        except Exception as e:
            print(f"  grpc-proxy :{grpc_port}{grpc_path}: {e}")
            try:
                self._send_error(502, f"gRPC proxy error: {e}")
            except Exception:
                pass

    def _serve_ws_relay(self, qs):
        """WebSocket reverse proxy: browser WSS ↔ ws://127.0.0.1:grpc_port/path.

        Rerun 0.30.x uses WebSocket (not HTTP/fetch) for the gRPC data stream.
        The JS WebSocket intercept redirects ws://HOST:GRPC_PORT/PATH to
        wss://HOST:MAIN_PORT/ws-relay?path=PATH so that all traffic stays on
        the same HTTPS server port and doesn't require a separate open port.

        Raw byte piping works because WebSocket frames are self-describing:
        - Browser→upstream frames include their masking key inline.
        - Upstream→browser frames are unmasked (server→client in WS spec).
        No frame parsing is needed; we just pipe bytes bidirectionally.
        """
        import base64
        import hashlib

        if self.headers.get("Upgrade", "").lower() != "websocket":
            self._send_error(400, "Expected WebSocket upgrade"); return

        active_port = self.server_state.get("_active_grpc_port", 0)
        if not active_port:
            self.send_response(503)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        ws_path = unquote((qs.get("path") or [self.server_state.get("_active_grpc_path", "/")])[0])
        # Strip the ?t=TIMESTAMP suffix added for unique-URL reconnection
        ws_path = ws_path.split("?")[0] or "/"
        ws_proto = self.headers.get("Sec-WebSocket-Protocol", "")

        # Complete WebSocket handshake with browser
        ws_key = self.headers.get("Sec-WebSocket-Key", "")
        accept = base64.b64encode(
            hashlib.sha1((ws_key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()
        ).decode()

        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        if ws_proto:
            self.send_header("Sec-WebSocket-Protocol", ws_proto.split(",")[0].strip())
        self.end_headers()
        self.wfile.flush()

        # Connect to upstream Rerun gRPC WebSocket server
        up_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            up_sock.connect(("127.0.0.1", active_port))
            up_key = base64.b64encode(b"genmanip-ws-proxy!!").decode()
            up_req = (
                f"GET {ws_path} HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{active_port}\r\n"
                f"Upgrade: websocket\r\n"
                f"Connection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {up_key}\r\n"
                f"Sec-WebSocket-Version: 13\r\n"
            )
            if ws_proto:
                up_req += f"Sec-WebSocket-Protocol: {ws_proto.split(',')[0].strip()}\r\n"
            up_req += "\r\n"
            up_sock.sendall(up_req.encode())

            # Read upstream 101 response
            up_buf = b""
            while b"\r\n\r\n" not in up_buf:
                chunk = up_sock.recv(4096)
                if not chunk:
                    raise RuntimeError("Upstream closed during WS handshake")
                up_buf += chunk
            if b"101" not in up_buf[:20]:
                raise RuntimeError(f"Upstream WS handshake failed: {up_buf[:200]!r}")
        except Exception as e:
            print(f"  [WS-RELAY] upstream connect failed :{active_port}{ws_path}: {e}")
            try:
                up_sock.close()
            except Exception:
                pass
            return

        print(f"  [WS-RELAY] connected :{active_port}{ws_path}")
        client_conn = self.connection  # SSL socket (already in WS framing mode)
        stop = threading.Event()

        def _pipe(src, dst):
            try:
                while not stop.is_set():
                    data = src.recv(4096)
                    if not data:
                        break
                    dst.sendall(data)
            except Exception:
                pass
            finally:
                stop.set()

        t1 = threading.Thread(target=_pipe, args=(client_conn, up_sock), daemon=True)
        t2 = threading.Thread(target=_pipe, args=(up_sock, client_conn), daemon=True)
        t1.start()
        t2.start()
        t1.join(timeout=600)
        t2.join(timeout=600)
        try:
            up_sock.close()
        except Exception:
            pass
        print(f"  [WS-RELAY] closed :{active_port}{ws_path}")

    def _serve_viewer(self, sub: str, query: str):
        """Serve cached Rerun viewer static files with COOP/COEP headers."""
        filename = sub.lstrip("/") if sub.lstrip("/") else "index.html"
        file_path = _VIEWER_DIR / filename
        if not file_path.exists() or not file_path.is_file():
            self._send_404(); return

        ext = file_path.suffix.lower()
        mime = {
            ".html": "text/html",
            ".js":   "application/javascript",
            ".wasm": "application/wasm",
        }.get(ext, "application/octet-stream")

        cache = "public, max-age=604800, immutable" if ext in (".wasm", ".js") else "no-cache"
        accept_enc = self.headers.get("Accept-Encoding", "")
        gz_path = Path(str(file_path) + ".gz")
        use_gz  = "gzip" in accept_enc and gz_path.exists() and ext in (".wasm", ".js")

        data = gz_path.read_bytes() if use_gz else file_path.read_bytes()

        if filename == "index.html":
            if use_gz:
                data = gzip.decompress(data)
                use_gz = False
            html = data.decode("utf-8")
            # Intercept Rerun's gRPC-web fetch calls.
            # Chrome forces h2 ALPN for URLs whose path looks like a gRPC service
            # path (e.g. /rerun.sdk_comms.v1alpha1.MessageProxyService/ReadMessages),
            # which breaks our HTTP/1.1-only TLS server.  We rewrite those fetches
            # to POST /relay (a plain path Chrome doesn't recognise as gRPC) and
            # carry the original path in an x-grpc-path header so do_POST can forward
            # it correctly to the Rerun gRPC server.
            # Rerun may substitute 127.0.0.1 with the page hostname before calling
            # fetch, so we detect by path content, not by hostname.
            patch = """
<script>
(function() {
    var _of = window.fetch;
    window.fetch = function(resource, init) {
        try {
            var u = resource instanceof Request ? resource.url : String(resource);
            var p = new URL(u);
            // Intercept any fetch that isn't same-origin (different host or port).
            // Rerun WASM substitutes the page hostname at runtime, so all non-same-
            // origin requests are Rerun internals (gRPC server, cloud catalog, etc.).
            // Rewrite to /relay to avoid Chrome's ERR_ALPN_NEGOTIATION_FAILED on
            // HTTPS + gRPC service path patterns.
            if (p.host && p.host !== window.location.host) {
                var origPath = p.pathname + p.search;
                var relay = window.location.protocol + '//' + window.location.host + '/relay';
                console.log('[relay] intercept', u, '→', relay, 'path:', origPath);
                var base = resource instanceof Request ? resource : null;
                var ni = base ? {
                    method: base.method,
                    headers: new Headers(base.headers),
                    body: base.body,
                    credentials: base.credentials,
                    cache: base.cache,
                    redirect: base.redirect,
                } : Object.assign({}, init || {});
                if (!(ni.headers instanceof Headers)) ni.headers = new Headers(ni.headers || {});
                ni.headers.set('x-grpc-path', origPath);
                // Buffer ReadableStream bodies into an ArrayBuffer before sending.
                // duplex:'half' streaming uploads are unreliable over HTTP/1.1 — our
                // server reads Content-Length bytes and gets 0 without buffering.
                // gRPC-web request frames are small so eager buffering is safe.
                if (ni.body && typeof ReadableStream !== 'undefined'
                        && ni.body instanceof ReadableStream) {
                    var reader = ni.body.getReader();
                    var chunks = [];
                    var pump = function() {
                        return reader.read().then(function(r) {
                            if (r.done) {
                                var total = 0;
                                chunks.forEach(function(c) { total += c.length; });
                                var buf = new Uint8Array(total), off = 0;
                                chunks.forEach(function(c) { buf.set(c, off); off += c.length; });
                                ni.body = buf.buffer;
                                return _of.call(window, relay, ni);
                            }
                            chunks.push(new Uint8Array(r.value));
                            return pump();
                        });
                    };
                    return pump();
                }
                resource = relay;
                init = ni;
            }
        } catch(e) {}
        return _of.call(this, resource, init);
    };
})();
// Intercept WebSocket connections to cross-origin hosts (Rerun data stream).
// Rerun 0.30.x uses WebSocket transport for gRPC; the viewer tries to open
// ws://HOST:GRPC_PORT/proxy which fails because GRPC_PORT is not accessible
// remotely.  We redirect to wss://MAIN_HOST:MAIN_PORT/ws-relay?path=PATH so
// all traffic goes through the same HTTPS server.
(function() {
    var _oWS = window.WebSocket;
    window.WebSocket = function(url, protocols) {
        try {
            var p = new URL(url);
            if (p.host && p.host !== window.location.host) {
                var wsScheme = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                var relay = wsScheme + '//' + window.location.host
                    + '/ws-relay?path=' + encodeURIComponent(p.pathname + p.search);
                console.log('[relay] WS intercept', url, '→', relay);
                return protocols ? new _oWS(relay, protocols) : new _oWS(relay);
            }
        } catch(e) { console.log('[relay] WS error', e); }
        return protocols ? new _oWS(url, protocols) : new _oWS(url);
    };
    window.WebSocket.CONNECTING = _oWS.CONNECTING;
    window.WebSocket.OPEN      = _oWS.OPEN;
    window.WebSocket.CLOSING   = _oWS.CLOSING;
    window.WebSocket.CLOSED    = _oWS.CLOSED;
    window.WebSocket.prototype = _oWS.prototype;
})();
// Expose _webhandle on the iframe window and notify the parent frame when
// Rerun's WebHandle is created.  The disk-patched index.html may not match
// Rerun 0.30.x's exact code, so we intercept wasm_bindgen assignment here
// instead.  The parent page's fetchAndLoad() needs both _webhandle (to call
// add_receiver) and the 'READY' message (to know the viewer is initialised).
(function() {
    function _installWebHandleHook(wb) {
        if (!wb || !wb.WebHandle || wb.WebHandle.__hooked) return;
        var _Orig = wb.WebHandle;
        function _Hooked() {
            var h = new _Orig(...arguments);
            window._webhandle = h;
            console.log('[genmanip] _webhandle set, posting READY');
            window.parent.postMessage('READY', '*');
            return h;
        }
        _Hooked.prototype = _Orig.prototype;
        _Hooked.__hooked = true;
        wb.WebHandle = _Hooked;
    }
    // Intercept wasm_bindgen assignment (set before WASM init runs).
    var _wbVal;
    Object.defineProperty(window, 'wasm_bindgen', {
        get: function() { return _wbVal; },
        set: function(v) { _wbVal = v; _installWebHandleHook(v); },
        configurable: true,
    });
    // Fallback: poll in case wasm_bindgen was already set.
    var _t = setInterval(function() {
        if (window.wasm_bindgen) { _installWebHandleHook(window.wasm_bindgen); }
        if (window._webhandle) clearInterval(_t);
    }, 200);
})();
</script>
"""
            html = html.replace("<head>", "<head>" + patch)
            data = html.encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache)
        if use_gz:
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (ssl.SSLError, BrokenPipeError, ConnectionResetError):
            pass

    # ---------- helpers ----------

    def _html(self, code: int, html: str, coep: bool = False):
        data = html.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        if coep:
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (ssl.SSLError, BrokenPipeError, ConnectionResetError):
            pass

    def _send_404(self):
        self._html(404, _page("Not Found", "<p class='info-msg'>404 – Not found.</p>"))

    def _send_error(self, code: int, msg: str):
        self._html(code, _page("Error", f"<p class='info-msg'>{code} – {msg}</p>"))


# ── Entry point ───────────────────────────────────────────────────────────────

def run_visualizer(project_root: Optional[str], port: int = 55077) -> None:
    root = Path(project_root).resolve() if project_root else Path.cwd()
    base = root / "saved" / "eval_results"

    runs = _find_runs(base)

    print(f"  Caching Rerun viewer assets...")
    _cache_viewer_assets()

    def _is_port_free(p):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", p))
                return True
            except OSError:
                return False

    state = {"runs": runs, "port": port, "scheme": "https", "_active_grpc_port": 0, "_active_grpc_path": "/"}

    class Handler(_Handler):
        server_state = state

    if not _is_port_free(port):
        for p in range(port + 1, port + 20):
            if _is_port_free(p):
                print(f"  Port {port} busy, using {p} instead.")
                port = p
                state["port"] = port
                break
        else:
            print(f"Error: no free port found near {port}")
            return

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)

    hostname = socket.gethostname()

    # HTTPS is required for WebCodecs (VideoDecoder API needs a secure context).
    # gRPC-web fetch calls are same-origin and reuse the existing h1.1 TLS
    # connection, so there is no ALPN renegotiation for the gRPC service paths.
    if not _ensure_cert(hostname):
        print("  Warning: could not create TLS certificate, falling back to HTTP.")
        state["scheme"] = "http"
    else:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.set_alpn_protocols(["http/1.1"])
        ctx.load_cert_chain(_CERT_FILE, _KEY_FILE)
        server.socket = ctx.wrap_socket(server.socket, server_side=True)

    scheme = state["scheme"]
    local_url = f"{scheme}://localhost:{port}/"
    host_url = f"{scheme}://{hostname}:{port}/"
    print(f"  Open locally: {local_url}")
    if hostname not in {"localhost", "127.0.0.1"}:
        print(f"  If running remotely, forward port {port} or open: {host_url}")
        print(f"  Example: ssh -L {port}:localhost:{port} <remote-host>")
    print(f"  Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")
    finally:
        server.server_close()
