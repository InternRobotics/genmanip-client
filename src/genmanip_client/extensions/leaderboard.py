from __future__ import annotations

import json
import os
import shutil
from typing import Any

import requests

DEFAULT_USER_TOKEN = os.getenv("USER_TOKEN", None)
DEFAULT_LEADERBOARD_HOST = os.getenv("LEADERBOARD_HOST", "localhost")
DEFAULT_LEADERBOARD_PORT = int(os.getenv("LEADERBOARD_PORT", 8000))


def resolve_project_root(project_root: str | None) -> str:
    if project_root:
        return os.path.abspath(project_root)
    return os.path.abspath(os.getcwd())


def upload_submission(
    host_ip: str,
    port: int,
    submission_file_path: str,
    author_token: str,
    submission_infos: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Upload a submission to the leaderboard server.
    """
    url = f"http://{host_ip}:{port}/api/upload"

    if not os.path.exists(submission_file_path):
        print(f"Error: File {submission_file_path} does not exist.")
        return None

    submission_name = submission_infos.get("submission_name")
    leaderboard_name = submission_infos.get("leaderboard_name")

    if not submission_name or not leaderboard_name:
        print(
            "Error: submission_infos must contain 'submission_name' and 'leaderboard_name'."
        )
        return None

    with open(submission_file_path, "rb") as submission_file:
        files = {
            "file": (
                os.path.basename(submission_file_path),
                submission_file,
                "application/zip",
            )
        }

    data = {
        "submission_name": submission_name,
        "leaderboard_name": leaderboard_name,
        "author_token": author_token,
        "submission_infos": json.dumps(submission_infos),
    }

    try:
        response = requests.post(url, files=files, data=data, timeout=(30, 300))
        response.raise_for_status()
        result = response.json()

        if result.get("status") == "success":
            print(
                f"Successfully uploaded! Submission ID: {result.get('submission_id')}"
            )
            print(
                f"Check status at: http://{host_ip}:{port}/upload_status/{result.get('submission_id')}"
            )
        else:
            print(f"Upload failed: {result.get('error')}")

        return result
    except (
        json.JSONDecodeError,
        requests.RequestException,
        RuntimeError,
        ValueError,
    ) as e:
        print(f"An error occurred: {e}")
        return {"status": "error", "error": str(e)}


def list_results(project_root: str | None) -> None:
    root = resolve_project_root(project_root)
    base_results_dir = os.path.join(root, "saved", "eval_results")
    results = []

    if os.path.exists(base_results_dir):
        for benchmark_id in os.listdir(base_results_dir):
            benchmark_dir = os.path.join(base_results_dir, benchmark_id)
            if not os.path.isdir(benchmark_dir):
                continue

            for run_id in os.listdir(benchmark_dir):
                run_dir = os.path.join(benchmark_dir, run_id)
                if os.path.isdir(run_dir):
                    submitted = os.path.exists(os.path.join(run_dir, "submitted.flag"))
                    results.append(
                        {
                            "benchmark_id": benchmark_id,
                            "run_id": run_id,
                            "path": run_dir,
                            "submitted": submitted,
                        }
                    )

    if not results:
        print("No results found.")
    else:
        GREEN = "\033[92m"
        RED = "\033[91m"
        RESET = "\033[0m"

        print(f"{'Benchmark ID':<30} {'Run ID':<40} {'Submitted':<15} {'Path'}")
        print("-" * 150)
        results = sorted(results, key=lambda x: (x["benchmark_id"], x["run_id"]))
        results.reverse()
        for item in results:
            submitted = item.get("submitted", False)
            status = f"{GREEN}Yes{RESET}" if submitted else f"{RED}No{RESET}"
            print(
                f"{item['benchmark_id']:<30} {item['run_id']:<40} {status:<24} {item['path']}"
            )


def submit_results(
    user_token: str,
    run_id: str,
    submission_name: str,
    leaderboard_name: str,
    host: str,
    port: int,
    project_root: str | None,
    benchmark_id: str | None = None,
) -> None:
    root = resolve_project_root(project_root)
    if benchmark_id:
        results_dir = os.path.join(root, "saved", "eval_results", benchmark_id, run_id)
    else:
        base_results_dir = os.path.join(root, "saved", "eval_results")
        results_dir = None
        if os.path.exists(base_results_dir):
            for b_id in os.listdir(base_results_dir):
                potential_dir = os.path.join(base_results_dir, b_id, run_id)
                if os.path.isdir(potential_dir):
                    results_dir = potential_dir
                    benchmark_id = b_id
                    break

    if results_dir is None or not os.path.exists(results_dir):
        print(f"Error: Results not found for run_id: {run_id}")
        return

    submission_info = {
        "submission_name": submission_name,
        "leaderboard_name": leaderboard_name,
    }

    print(f"Submitting run {run_id} to {leaderboard_name} at {host}:{port}...")

    zip_filename = f"results_{run_id}"
    zip_base_path = os.path.join(os.path.dirname(results_dir), zip_filename)
    archive_path = shutil.make_archive(zip_base_path, "zip", results_dir)

    try:
        result = upload_submission(
            host,
            port,
            archive_path,
            user_token,
            submission_info,
        )

        if result and result.get("status") == "success":
            submitted_flag_path = os.path.join(results_dir, "submitted.flag")
            with open(submitted_flag_path, "w") as f:
                json.dump(submission_info, f, indent=4)
        else:
            print("Submission failed.")

    finally:
        if os.path.exists(archive_path):
            os.remove(archive_path)
