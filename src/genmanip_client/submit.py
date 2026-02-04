"""Submit command for starting evaluation jobs on the GenManip eval server."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

import requests

DEFAULT_TIMEOUT = 30.0

# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    # Foreground
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    # Bright foreground
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_CYAN = "\033[96m"


# Box drawing characters
class Box:
    TL = "╭"  # top-left
    TR = "╮"  # top-right
    BL = "╰"  # bottom-left
    BR = "╯"  # bottom-right
    H = "─"   # horizontal
    V = "│"   # vertical
    LT = "├"  # left-tee
    RT = "┤"  # right-tee


def colored(text: str, *colors: str) -> str:
    """Wrap text with ANSI color codes."""
    if not sys.stdout.isatty():
        return text
    return "".join(colors) + text + Colors.RESET


def make_box_line(content: str, width: int, left: str = Box.V, right: str = Box.V) -> str:
    """Create a box line with content padded to width."""
    padding = width - len(content)
    return f"{left} {content}{' ' * padding} {right}"


def make_progress_bar(completed: int, total: int, width: int = 20) -> str:
    """Create a visual progress bar."""
    if total == 0:
        ratio = 0.0
    else:
        ratio = completed / total
    filled = int(width * ratio)
    empty = width - filled
    bar = "█" * filled + "░" * empty
    percent = ratio * 100
    return f"{bar} {percent:5.1f}%"


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
    width = 52
    inner_width = width - 4  # account for box chars and spaces

    # Status styling
    status_val = status.get("status", "unknown")
    status_colors = {
        "running": (Colors.BRIGHT_YELLOW, "⟳"),
        "complete": (Colors.BRIGHT_GREEN, "✓"),
        "incomplete": (Colors.YELLOW, "○"),
        "idle": (Colors.DIM, "◦"),
    }
    color, icon = status_colors.get(status_val, (Colors.WHITE, "?"))

    # Header
    print(colored(f"{Box.TL}{Box.H * (width - 2)}{Box.TR}", Colors.CYAN))
    title = "GenManip Evaluation Status"
    title_pad = (inner_width - len(title)) // 2
    print(colored(Box.V, Colors.CYAN) + " " + " " * title_pad + colored(title, Colors.BOLD, Colors.BRIGHT_CYAN) + " " * (inner_width - title_pad - len(title)) + " " + colored(Box.V, Colors.CYAN))
    print(colored(f"{Box.LT}{Box.H * (width - 2)}{Box.RT}", Colors.CYAN))

    # Status info
    status_display = colored(f"{icon} {status_val.upper()}", Colors.BOLD, color)
    print(colored(Box.V, Colors.CYAN) + f"  {'Status:':<14} {status_display}" + " " * (inner_width - 15 - len(status_val) - 3) + " " + colored(Box.V, Colors.CYAN))

    benchmark = status.get("benchmark_id", "N/A") or "N/A"
    print(colored(Box.V, Colors.CYAN) + f"  {'Benchmark:':<14} {colored(benchmark, Colors.WHITE)}" + " " * (inner_width - 15 - len(str(benchmark))) + " " + colored(Box.V, Colors.CYAN))

    run_id = status.get("run_id", "N/A") or "N/A"
    run_id_display = str(run_id)[:30] + "..." if len(str(run_id)) > 30 else str(run_id)
    print(colored(Box.V, Colors.CYAN) + f"  {'Run ID:':<14} {colored(run_id_display, Colors.DIM)}" + " " * (inner_width - 15 - len(run_id_display)) + " " + colored(Box.V, Colors.CYAN))

    # Separator
    print(colored(f"{Box.LT}{Box.H * (width - 2)}{Box.RT}", Colors.CYAN))

    # Progress
    total = status.get("total_episodes", 0)
    completed = status.get("completed_episodes", 0)
    in_progress = status.get("in_progress_episodes", 0)

    progress_bar = make_progress_bar(completed, total)
    progress_text = f"{completed}/{total}"
    print(colored(Box.V, Colors.CYAN) + f"  {'Progress:':<14} {colored(progress_bar, Colors.GREEN)}" + " " * (inner_width - 15 - 28) + " " + colored(Box.V, Colors.CYAN))
    print(colored(Box.V, Colors.CYAN) + f"  {'Completed:':<14} {colored(str(completed), Colors.BRIGHT_GREEN)}" + " " * (inner_width - 15 - len(str(completed))) + " " + colored(Box.V, Colors.CYAN))
    print(colored(Box.V, Colors.CYAN) + f"  {'In Progress:':<14} {colored(str(in_progress), Colors.YELLOW)}" + " " * (inner_width - 15 - len(str(in_progress))) + " " + colored(Box.V, Colors.CYAN))

    workers = status.get("active_workers", [])
    workers_str = ", ".join(map(str, workers)) if workers else colored("none", Colors.DIM)
    workers_display = workers_str[:25] + "..." if len(workers_str) > 25 else workers_str
    raw_len = len(", ".join(map(str, workers))) if workers else 4
    display_len = min(raw_len, 25) + (3 if raw_len > 25 else 0)
    print(colored(Box.V, Colors.CYAN) + f"  {'Workers:':<14} {workers_display}" + " " * (inner_width - 15 - display_len) + " " + colored(Box.V, Colors.CYAN))

    # Results section
    results = status.get("results", {})
    print(colored(f"{Box.LT}{Box.H * (width - 2)}{Box.RT}", Colors.CYAN))

    if results:
        results_title = "Results (Success Rate)"
        print(colored(Box.V, Colors.CYAN) + f"  {colored(results_title, Colors.BOLD)}" + " " * (inner_width - 1 - len(results_title)) + " " + colored(Box.V, Colors.CYAN))
        print(colored(Box.V, Colors.CYAN) + " " * (inner_width + 2) + " " + colored(Box.V, Colors.CYAN))
        for task_name, sr in sorted(results.items()):
            sr_color = Colors.BRIGHT_GREEN if sr >= 0.8 else Colors.YELLOW if sr >= 0.5 else Colors.RED
            sr_bar_width = 15
            sr_filled = int(sr_bar_width * sr)
            sr_bar = colored("▓" * sr_filled, sr_color) + colored("░" * (sr_bar_width - sr_filled), Colors.DIM)
            task_display = task_name[:20] + ".." if len(task_name) > 20 else task_name
            sr_str = f"{sr:.2%}"
            line_content = f"  {task_display:<22} {sr_bar} {colored(sr_str, sr_color, Colors.BOLD)}"
            # Calculate raw length for padding
            raw_task_len = min(len(task_name), 20) + (2 if len(task_name) > 20 else 0)
            pad = inner_width - 2 - 22 - sr_bar_width - 1 - 7
            print(colored(Box.V, Colors.CYAN) + line_content + " " * max(0, pad) + " " + colored(Box.V, Colors.CYAN))
    else:
        no_results = "No results yet"
        print(colored(Box.V, Colors.CYAN) + f"  {colored(no_results, Colors.DIM)}" + " " * (inner_width - 1 - len(no_results)) + " " + colored(Box.V, Colors.CYAN))

    # Footer
    print(colored(f"{Box.BL}{Box.H * (width - 2)}{Box.BR}", Colors.CYAN))


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


def print_error(message: str) -> None:
    """Print an error message with styling."""
    icon = colored("✗", Colors.RED, Colors.BOLD)
    print(f"{icon} {colored('Error:', Colors.RED, Colors.BOLD)} {message}", file=sys.stderr)


def print_warning(message: str) -> None:
    """Print a warning message with styling."""
    icon = colored("⚠", Colors.YELLOW)
    print(f"{icon} {colored(message, Colors.YELLOW)}")


def print_info(message: str) -> None:
    """Print an info message with styling."""
    icon = colored("→", Colors.CYAN)
    print(f"{icon} {message}")


def print_success(message: str) -> None:
    """Print a success message with styling."""
    icon = colored("✓", Colors.BRIGHT_GREEN, Colors.BOLD)
    print(f"{icon} {colored(message, Colors.BRIGHT_GREEN)}")


def print_hint(message: str) -> None:
    """Print a hint/suggestion with styling."""
    print(f"  {colored('↳', Colors.DIM)} {colored(message, Colors.DIM)}")


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
        print_error(str(e))
        return 1

    current_status = status.get("status", "idle")
    active_workers = status.get("active_workers", [])
    benchmark_id = status.get("benchmark_id")
    previous_run_id = status.get("run_id")

    # Handle different status scenarios
    if current_status == "running" and active_workers:
        # Case 1: Workers are still active - must kill them first
        print_error(f"Job is running with active workers: {colored(str(active_workers), Colors.YELLOW)}")
        print_hint(f"Benchmark: {benchmark_id}, Run ID: {previous_run_id}")
        print_hint("Please kill the workers first before submitting a new job.")
        return 1

    elif current_status == "incomplete" and not active_workers:
        # Case 2: Previous job incomplete, no active workers
        if args.run_id == previous_run_id:
            # Resuming the same run
            print_info(f"Resuming run: {colored(previous_run_id, Colors.CYAN, Colors.BOLD)}")
        else:
            # Starting a new run, warn about previous incomplete job
            print_warning(f"Previous job was incomplete ({benchmark_id} / {previous_run_id})")
            print_hint(
                f"Completed: {status.get('completed_episodes', 0)} / "
                f"{status.get('total_episodes', 0)}"
            )
            print_hint(f"To resume: {colored(f'gmp submit <config> --run-id {previous_run_id}', Colors.CYAN)}")
            print_hint(
                f"To re-test: manually clean {colored(f'saved/eval_results/{benchmark_id}/{previous_run_id}/', Colors.DIM)}"
            )
            print()

    elif current_status == "complete":
        # Case 3: Previous job completed
        if args.run_id == previous_run_id:
            # User wants to see previous results
            print_success(f"Run '{previous_run_id}' already completed:")
            print_status(status)
            return 0
        else:
            # Starting a new run
            print_info(f"Previous job completed ({benchmark_id} / {previous_run_id})")
            print_hint(f"To view results: {colored(f'gmp submit <config> --run-id {previous_run_id}', Colors.CYAN)}")
            print()

    # Determine run_id
    effective_run_id = args.run_id
    if effective_run_id is None:
        effective_run_id = datetime.now().strftime("%Y-%m-%d_%H_%M_%S_%f")
        print_info(f"Generated run_id: {colored(effective_run_id, Colors.CYAN)}")

    # Start the job
    try:
        print_info(f"Starting job with config(s): {colored(str(args.config_paths), Colors.WHITE)}")
        start_job(base_url, args.config_paths, effective_run_id)
        print_success("Job started successfully!")
        print()

        # Get and print updated status
        new_status = get_server_status(base_url)
        print_status(new_status)
        return 0

    except RuntimeError as e:
        print_error(str(e))
        return 1


def main(argv: list[str] | None = None) -> int:
    """Main entry point for submit command."""
    parser = build_argparser()
    args = parser.parse_args(argv)
    return run_submit(args)
