import pickle

import pytest
import requests

from genmanip_client.reliable_transport import PROTOCOL, ReliableTransport


def response(data, status=200):
    result = requests.Response()
    result.status_code = status
    result._content = pickle.dumps(data)
    return result


class Session:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = []

    def get(self, *args, **kwargs):
        result = requests.Response()
        result.status_code = 200
        result._content = b'{"protocol":"genmanip-replay-v1","epoch":"server-1","worker_generations":{}}'
        return result

    def post(self, *args, **kwargs):
        self.calls.append(kwargs)
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, requests.Response):
            return outcome
        envelope = pickle.loads(kwargs["data"])
        result = {k: envelope[k] for k in ("epoch", "stream_id", "sequence")}
        result.update(
            protocol=PROTOCOL, state="completed", status_code=200, body=b"observation"
        )
        if isinstance(outcome, dict):
            result.update(outcome)
        if isinstance(outcome, int):
            return response({}, outcome)
        return response(result, 202 if result["state"] == "pending" else 200)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("genmanip_client.reliable_transport.time.sleep", lambda _: None)


def transport(session, **kwargs):
    return ReliableTransport(
        "http://example",
        ["0"],
        {"Authorization": "Bearer test"},
        {"run_id": "test-run"},
        session=session,
        **kwargs,
    )


def test_lost_requests_responses_and_pending_keep_identical_id_and_body():
    session = Session(
        [
            requests.ConnectTimeout(),
            requests.ReadTimeout(),
            450,
            502,
            {"state": "pending"},
            {},
            {},
        ]
    )
    client = transport(session)
    assert client.post("step_chunk", b"actions", 100).content == b"observation"
    assert len({call["data"] for call in session.calls}) == 1
    assert all(call["timeout"] == 10 for call in session.calls)
    assert session.calls[0]["params"] == {"run_id": "test-run"}
    assert session.calls[0]["headers"]["Authorization"] == "Bearer test"
    client.post("step_chunk", b"next-actions", 100)
    assert pickle.loads(session.calls[-1]["data"])["sequence"] == 2


@pytest.mark.parametrize(
    "result",
    [
        {"state": "rejected", "detail": "Server epoch changed"},
        {"sequence": 99},
        {"status_code": 500, "detail": "Worker died"},
    ],
)
def test_terminal_error_is_not_retried_and_prevents_next_action(result):
    session = Session([result])
    client = transport(session)
    with pytest.raises(RuntimeError):
        client.post("step_chunk", b"actions", 100)
    with pytest.raises(RuntimeError, match="explicitly reset"):
        client.post("step_chunk", b"other-actions", 100)
    assert len(session.calls) == 1


def test_old_server_is_rejected_before_any_action():
    session = Session([])
    session.get = lambda *a, **k: response({}, 404)
    with pytest.raises(RuntimeError, match="require a server"):
        transport(session).post("step_chunk", b"actions", 100)
    assert session.calls == []


def test_network_silence_budget_stops_retrying(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(
        "genmanip_client.reliable_transport.time.monotonic", lambda: clock[0]
    )
    monkeypatch.setattr(
        "genmanip_client.reliable_transport.time.sleep",
        lambda n: clock.__setitem__(0, clock[0] + n),
    )
    session = Session([requests.ReadTimeout()] * 10)
    client = transport(session, recovery_timeout=3)
    with pytest.raises(RuntimeError, match="network recovery deadline"):
        client.post("step_chunk", b"actions", 100)
    assert len(session.calls) == 3


def test_pending_refreshes_network_budget_but_not_operation_deadline(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(
        "genmanip_client.reliable_transport.time.monotonic", lambda: clock[0]
    )
    monkeypatch.setattr(
        "genmanip_client.reliable_transport.time.sleep",
        lambda n: clock.__setitem__(0, clock[0] + n),
    )
    session = Session([{"state": "pending"}] * 10)
    client = transport(session, recovery_timeout=2)
    with pytest.raises(RuntimeError, match="operation or network recovery deadline"):
        client.post("step_chunk", b"actions", 5)
    assert len(session.calls) == 5


def test_reset_starts_a_new_stream():
    client = transport(Session([{}, {}]))
    client.post("reset", b"workers", 100)
    old_id = client.stream_id
    client.new_stream()
    client.post("reset", b"workers", 100)
    assert client.sequence == 1
    assert client.stream_id != old_id


@pytest.mark.parametrize(
    "status,body",
    [
        (450, b"Internal Server Error"),
        (500, b"Internal Server Error"),
        (502, b"<html>Bad Gateway</html>"),
        (504, b""),
        (200, b"<html>Gateway Error</html>"),
        (200, b""),
        (200, b'\x80\x05\x95'),
    ],
)
def test_non_json_gateway_errors_and_truncated_responses_are_retried(status, body):
    gateway_response = requests.Response()
    gateway_response.status_code = status
    gateway_response._content = body
    session = Session([gateway_response, {}])
    client = transport(session)
    assert client.post("step_chunk", b"actions", 100).content == b"observation"
    assert len(session.calls) == 2
    assert session.calls[0]["data"] == session.calls[1]["data"]


def test_non_json_discovery_response_reports_clear_error_before_actions():
    gateway_response = requests.Response()
    gateway_response.status_code = 200
    gateway_response._content = b"<html>Gateway Error</html>"
    session = Session([])
    session.get = lambda *args, **kwargs: gateway_response
    with pytest.raises(RuntimeError, match="Invalid reliable request capability response"):
        transport(session).post("step_chunk", b"actions", 100)
    assert session.calls == []


def test_eval_client_uses_replay_by_default_without_new_arguments(monkeypatch, tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    pytest.importorskip("filelock")
    from genmanip_client.eval_client import EvalClient

    monkeypatch.setenv("GENMANIP_RESULT_DIR", str(tmp_path))
    monkeypatch.setattr(EvalClient, "_health_check", lambda self: None)
    client = EvalClient("http://example", save_result=False, save_process=False)
    transport = client._reliable_transport
    transport.session.close()
    session = Session([requests.ReadTimeout(), {}])
    transport.session = session
    assert client._post_operation("step_chunk", b"actions", 100).content == b"observation"
    assert session.calls[0]["data"] == session.calls[1]["data"]
    assert session.calls[0]["timeout"] == 10
