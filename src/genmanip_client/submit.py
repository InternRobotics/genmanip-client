"""Submit command for starting evaluation jobs on the GenManip eval server."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

import requests

DEFAULT_TIMEOUT = 30.0


def get_server_status(base_url: str, timeout: float = DEFAULT_TIMEOUT) -> dict:
    """
    Get current server status.

    Args:
        base_url: Server base URL
        timeout: Request timeout in seconds

    Returns:
        Status dict from server

    Raises:
        RuntimeError: If server is not reachable or returns error
    """
    try:
        resp = requests.get(f"{base_url}/status", timeout=timeout)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Server returned status {resp.status_code}: {resp.text}"
            )
        data = resp.json()
        return data.get("data", {})
    except requests.ConnectionError as e:
        raise RuntimeError(
            f"Cannot connect to server at {base_url}. "
            f"Ensure the evaluation server is running. Error: {e}"
        ) from e
    except requests.Timeout:
        raise RuntimeError(f"Server request timed out after {timeout}s")


def start_job(
    base_url: str,
    config_paths: list[str],
    run_id: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> None:
    """
    Start a new evaluation job on the server.

    Args:
        base_url: Server base URL
        config_paths: List of config file paths
        run_id: Optional run ID (auto-generated if not provided)
        timeout: Request timeout in seconds

    Raises:
        RuntimeError: If request fails
    """
    payload = {
        "data": {
            "config_path": config_paths,
            "run_id": run_id,
        }
    }
    try:
        resp = requests.post(
            f"{base_url}/start_new_job",
            json=payload,
            timeout=timeout,
        )
        if resp.status_code != 200:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise RuntimeError(
                f"Failed to start job: {resp.status_code} - {detail}"
            )
    except requests.ConnectionError as e:
        raise RuntimeError(f"Cannot connect to server: {e}") from e
    except requests.Timeout:
        raise RuntimeError(f"Start job request timed out after {timeout}s")


def print_status(status: dict) -> None:
    """Print formatted status information."""
    print("=" * 50)
    print(f"Status: {status.get('status', 'unknown')}")
    print(f"Benchmark: {status.get('benchmark_id', 'N/A')}")
    print(f"Run ID: {status.get('run_id', 'N/A')}")
    print("-" * 50)
    print(f"Total episodes: {status.get('total_episodes', 0)}")
    print(f"Completed: {status.get('completed_episodes', 0)}")
    print(f"In progress: {status.get('in_progress_episodes', 0)}")
    print(f"Active workers: {status.get('active_workers', [])}")
    print("-" * 50)
    results = status.get("results", {})
    if results:
        print("Results:")
        for task_name, sr in sorted(results.items()):
            print(f"  {task_name}: {sr:.4f}")
    else:
        print("Results: (none yet)")
    print("=" * 50)


def build_argparser() -> argparse.ArgumentParser:
    """Build argument parser for submit command."""
    parser = argparse.ArgumentParser(
        prog="gmp submit",
        description="Submit evaluation jobs to GenManip eval server",
    )
    parser.add_argument(
        "config_paths",
        nargs="+",
        help="Config file path(s) to evaluate",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID for this evaluation (auto-generated if not provided)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force restart even if a job is in progress or completed",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8087,
        help="Server port (default: 8087)",
    )
    return parser


def run_submit(args: argparse.Namespace) -> int:
    """
    Run the submit command.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    base_url = f"http://{args.host}:{args.port}"

    try:
        # Get current server status
        status = get_server_status(base_url)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    current_status = status.get("status", "idle")
    active_workers = status.get("active_workers", [])
    benchmark_id = status.get("benchmark_id")
    run_id = status.get("run_id")

    # Handle different status scenarios
    if current_status == "running" and active_workers:
        if not args.overwrite:
            print(
                f"Error: Job already in progress with workers {active_workers}.",
                file=sys.stderr,
            )
            print(
                f"  Benchmark: {benchmark_id}, Run ID: {run_id}",
                file=sys.stderr,
            )
            print(
                "Use --overwrite to force restart the job.",
                file=sys.stderr,
            )
            return 1
        print(f"Warning: Overwriting running job (workers: {active_workers})")

    elif current_status == "incomplete" and not active_workers:
        if not args.overwrite and not args.run_id:
            print(
                f"Previous job incomplete: {benchmark_id} / {run_id}",
                file=sys.stderr,
            )
            print(
                f"  Completed: {status.get('completed_episodes', 0)} / "
                f"{status.get('total_episodes', 0)}",
                file=sys.stderr,
            )
            print("Options:", file=sys.stderr)
            print(
                f"  --run-id {run_id}  : Resume this run",
                file=sys.stderr,
            )
            print("  --overwrite       : Start fresh", file=sys.stderr)
            return 1
        if args.run_id == run_id:
            print(f"Resuming run: {run_id}")

    elif current_status == "complete":
        if not args.overwrite:
            print("Previous job completed:")
            print_status(status)
            print(
                "\nUse --overwrite to restart the job.",
                file=sys.stderr,
            )
            return 0
        print("Warning: Overwriting completed job")

    # Determine run_id
    effective_run_id = args.run_id
    if effective_run_id is None:
        effective_run_id = datetime.now().strftime("%Y-%m-%d_%H_%M_%S_%f")
        print(f"Generated run_id: {effective_run_id}")

    # Start the job
    try:
        print(f"Starting job with config(s): {args.config_paths}")
        start_job(base_url, args.config_paths, effective_run_id)
        print("Job started successfully.")

        # Get and print updated status
        new_status = get_server_status(base_url)
        print_status(new_status)
        return 0

    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    """Main entry point for submit command."""
    parser = build_argparser()
    args = parser.parse_args(argv)
    return run_submit(args)
