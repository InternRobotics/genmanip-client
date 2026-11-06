"""Ordered replay transport; requires genmanip-replay-v1 on the server."""

import logging
import math
import pickle
import threading
import time
import uuid

import requests

logger = logging.getLogger(__name__)
PROTOCOL = "genmanip-replay-v1"


class ReliableTransport:
    def __init__(
        self,
        base_url,
        worker_ids,
        headers,
        params,
        request_timeout=10.0,
        recovery_timeout=60.0,
        session=None,
    ):
        if any(
            not math.isfinite(v) or v <= 0 for v in (request_timeout, recovery_timeout)
        ):
            raise ValueError(
                "Request and recovery timeouts must be finite and positive"
            )
        self.base_url = base_url.rstrip("/")
        self.worker_ids = list(map(str, worker_ids))
        self.headers, self.params = headers, params
        self.request_timeout = request_timeout
        self.recovery_timeout = recovery_timeout
        self.session = session if session is not None else requests.Session()
        self._lock = threading.Lock()
        self.new_stream()

    def new_stream(self):
        """Only call after explicitly killing/recreating the previous workers."""
        self.epoch = None
        self.stream_id = uuid.uuid4().hex
        self.sequence = 0
        self.failed = False

    def close(self):
        self.session.close()

    def _negotiate(self):
        response = self.session.get(
            self.base_url + "/request_protocol",
            headers=self.headers,
            params=self.params,
            timeout=self.request_timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(
                "Reliable requests require a server with genmanip-replay-v1; "
                f"protocol discovery returned HTTP {response.status_code}"
            )
        try:
            protocol = response.json()
        except ValueError as exc:
            raise RuntimeError("Invalid reliable request capability response") from exc
        if (
            not isinstance(protocol, dict)
            or protocol.get("protocol") != PROTOCOL
            or not protocol.get("epoch")
        ):
            raise RuntimeError("Server does not support genmanip-replay-v1")
        self.epoch = protocol["epoch"]
        generations = protocol.get("worker_generations")
        if not isinstance(generations, dict):
            raise RuntimeError("Server capability response has no worker generations")
        self.worker_generations = {w: generations.get(w, 0) for w in self.worker_ids}

    def post(self, operation, payload, timeout):
        # A second caller must not evict an observation before its first caller reads it.
        with self._lock:
            if self.failed:
                raise RuntimeError(
                    "Previous reliable request failed; explicitly reset before sending more actions"
                )
            if self.epoch is None:
                self._negotiate()  # Fail closed: never fall back to a non-idempotent POST.
            self.sequence += 1
            envelope = dict(
                epoch=self.epoch,
                stream_id=self.stream_id,
                sequence=self.sequence,
                operation=operation,
                worker_ids=self.worker_ids,
                payload=payload,
                worker_generations=self.worker_generations,
            )
            wire = pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)
            request_id = f"{self.stream_id}:{self.sequence}"
            deadline = time.monotonic() + timeout
            last_contact = time.monotonic()
            attempts = 0
            try:
                while True:
                    now = time.monotonic()
                    remaining = min(
                        deadline - now, self.recovery_timeout - (now - last_contact)
                    )
                    if remaining <= 0:
                        raise RuntimeError(
                            f"Reliable request {request_id} exceeded its operation or network recovery deadline; "
                            "execution may have completed on the server"
                        )
                    attempts += 1
                    started = time.monotonic()
                    try:
                        response = self.session.post(
                            self.base_url + "/reliable_request",
                            data=wire,
                            headers={
                                **self.headers,
                                "Content-Type": "application/octet-stream",
                            },
                            params=self.params,
                            timeout=min(self.request_timeout, remaining),
                        )
                        if response.status_code in (408, 429, 450, 500, 502, 503, 504):
                            raise requests.ConnectionError(
                                f"Transient HTTP {response.status_code}: {response.text[:200]}"
                            )
                        if response.status_code not in (200, 202):
                            raise RuntimeError(
                                f"Reliable request HTTP {response.status_code}: {response.text[:200]}"
                            )
                        try:
                            result = pickle.loads(response.content)
                        except (
                            EOFError,
                            ValueError,
                            TypeError,
                            pickle.UnpicklingError,
                        ) as exc:
                            raise requests.ConnectionError(
                                "Incomplete or invalid replay response"
                            ) from exc
                        if (
                            not isinstance(result, dict)
                            or result.get("protocol") != PROTOCOL
                        ):
                            raise RuntimeError("Invalid reliable response protocol")
                        if result.get("state") == "rejected":
                            raise RuntimeError(
                                f"Reliable request {request_id} rejected: {result.get('detail')}"
                            )
                        if any(
                            result.get(k) != envelope[k]
                            for k in ("epoch", "stream_id", "sequence")
                        ):
                            raise RuntimeError(
                                "Reliable response ID does not match the request"
                            )
                        last_contact = time.monotonic()
                        if result.get("state") == "completed":
                            if result.get("status_code") != 200:
                                raise RuntimeError(
                                    f"Server operation {operation} failed for {request_id}: "
                                    f"HTTP {result.get('status_code')} - {result.get('detail')}"
                                )
                            if not isinstance(result.get("body"), bytes):
                                raise RuntimeError(
                                    "Reliable response has no binary result"
                                )
                            response = requests.Response()
                            response.status_code = 200
                            response._content = result["body"]
                            logger.info(
                                "Reliable request completed id=%s operation=%s attempts=%s",
                                request_id,
                                operation,
                                attempts,
                            )
                            return response
                        if result.get("state") != "pending":
                            raise RuntimeError("Invalid reliable response state")
                        logger.debug(
                            "Reliable request pending id=%s operation=%s",
                            request_id,
                            operation,
                        )
                    except (
                        requests.Timeout,
                        requests.ConnectionError,
                        requests.exceptions.ChunkedEncodingError,
                    ) as exc:
                        logger.warning(
                            "Retrying reliable request id=%s operation=%s attempt=%s: %s",
                            request_id,
                            operation,
                            attempts,
                            exc,
                        )
                    # Avoid a tight retry loop on a fast proxy error or immediate pending.
                    delay = min(
                        max(0.0, 1.0 - (time.monotonic() - started)),
                        max(0.0, deadline - time.monotonic()),
                    )
                    if delay:
                        time.sleep(delay)
            except BaseException:
                # Never issue the next sequence after an ambiguous outcome.
                self.failed = True
                raise
