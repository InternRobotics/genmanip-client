import pytest

from genmanip_client.online_client import OnlineEvaluationClient


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def json(self):
        if isinstance(self._json_data, Exception):
            raise self._json_data
        return self._json_data


class FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append(
            {
                "url": url,
                "json": json,
                "headers": headers,
                "timeout": timeout,
            }
        )
        if not self._responses:
            raise AssertionError("No more fake responses configured")
        return self._responses.pop(0)


def test_create_task_sends_token_and_payload():
    session = FakeSession(
        [
            FakeResponse(
                json_data={"code": 0, "data": {"task_id": "T1"}},
                text="ok",
            )
        ]
    )
    client = OnlineEvaluationClient(
        base_url="https://example.com",
        token="TOKEN",
        session=session,
    )
    resp = client.create_task(task_id="T1")
    assert resp["data"]["task_id"] == "T1"
    assert len(session.calls) == 1
    call = session.calls[0]
    assert (
        call["url"]
        == "https://example.com/api/v1/benchmark/onlineEvaluation/createTask"
    )
    assert call["json"] == {"task_id": "T1"}
    assert call["headers"] == {"Authorization": "Bearer TOKEN"}


def test_ready_without_token():
    session = FakeSession(
        [
            FakeResponse(
                json_data={"code": 0, "data": {"ready": False}},
                text="ok",
            )
        ]
    )
    client = OnlineEvaluationClient(
        base_url="https://example.com",
        token=None,
        session=session,
    )
    resp = client.ready(task_id="T2")
    assert "data" in resp
    assert session.calls[0]["headers"] == {}


def test_wait_until_ready_polls(monkeypatch):
    session = FakeSession(
        [
            FakeResponse(json_data={"code": 0, "data": {"ready": False}}),
            FakeResponse(
                json_data={
                    "code": 0,
                    "data": {"ready": True, "endpoint": "http://x"},
                }
            ),
        ]
    )
    monkeypatch.setattr(
        "genmanip_client.online_client.time.sleep",
        lambda _: None,
    )
    client = OnlineEvaluationClient(
        base_url="https://example.com",
        token="TOKEN",
        session=session,
    )
    resp = client.wait_until_ready(
        task_id="T3",
        interval=0.0,
        timeout=None,
        pretty=False,
    )
    assert resp["data"]["ready"] is True
    assert len(session.calls) == 2
