from __future__ import annotations

import time

import requests

DEFAULT_TIMEOUT = 30.0
DEFAULT_ONLINE_EVAL_CREATE_TASK_PATH = "/api/v1/benchmark/onlineEvaluation/createTask"
DEFAULT_ONLINE_EVAL_READY_PATH = "/api/v1/benchmark/onlineEvaluation/ready"

class OnlineEvaluationClient:
    """Client for online evaluation API."""

    def __init__(
        self,
        base_url: str,
        token: str | None,
        timeout: float = DEFAULT_TIMEOUT,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.session = session or requests.Session()

    def _headers(self) -> dict[str, str]:
        if self.token:
            return {"Authorization": f"Bearer {self.token}"}
        return {}

    def _post(self, path: str, payload: dict) -> dict:
        try:
            resp = self.session.post(
                f"{self.base_url}{path}",
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Request failed: {exc}") from exc

        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

        try:
            data = resp.json()
        except ValueError as exc:
            raise RuntimeError(f"Invalid JSON response: {resp.text}") from exc

        if isinstance(data, dict) and data.get("code", 0) != 0:
            msg = data.get("msg", "")
            raise RuntimeError(f"API error {data.get('code')}: {msg}")

        return data

    def create_task(
        self,
        task_id: str | None = None,
        model_name: str | None = None,
        model_type: str | None = None,
        benchmark_set: str | None = None,
        submitter_name: str | None = None,
        submitter_homepage: str | None = None,
        is_public: int | None = None,
    ) -> dict:
        payload: dict[str, str | int] = {}
        if task_id:
            payload["task_id"] = task_id
        if model_name:
            payload["model_name"] = model_name
        if model_type:
            payload["model_type"] = model_type
        if benchmark_set:
            payload["benchmark_set"] = benchmark_set
        if submitter_name:
            payload["submitter_name"] = submitter_name
        if submitter_homepage:
            payload["submitter_homepage"] = submitter_homepage
        if is_public is not None:
            payload["is_public"] = is_public
        return self._post(
            DEFAULT_ONLINE_EVAL_CREATE_TASK_PATH,
            payload,
        )

    def ready(self, task_id: str) -> dict:
        payload = {"task_id": task_id}
        return self._post(
            DEFAULT_ONLINE_EVAL_READY_PATH,
            payload,
        )

    def wait_until_ready(
        self,
        task_id: str,
        interval: float = 5.0,
        timeout: float | None = None,
        pretty: bool = True,
    ) -> dict:
        start = time.time()
        if pretty:
            print(f"Waiting for available server (task_id={task_id})...")
        while True:
            resp = self.ready(task_id=task_id)
            data = resp.get("data", {}) if isinstance(resp, dict) else {}
            if isinstance(data, dict) and data.get("ready") is True:
                if pretty:
                    elapsed = time.time() - start
                    endpoint = data.get("endpoint")
                    endpoint_msg = f" endpoint={endpoint}" if endpoint else ""
                    print(
                        f"Ready after {elapsed:.1f}s.{endpoint_msg}"
                    )
                return resp
            if timeout is not None and (time.time() - start) >= timeout:
                raise RuntimeError(f"Timed out after {timeout}s waiting for ready")
            if pretty:
                elapsed = time.time() - start
                print(
                    f"Still waiting... elapsed {elapsed:.1f}s. Next check in {interval:.1f}s."
                )
            time.sleep(interval)
