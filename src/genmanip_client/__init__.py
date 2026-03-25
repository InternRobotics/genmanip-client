__all__ = [
    "EvalClient",
    "deserialize_data",
    "fake_action",
    "StreamingEpisodeRecorder",
]


def __getattr__(name: str):
    if name in {"EvalClient", "deserialize_data", "fake_action"}:
        from .eval_client import EvalClient, deserialize_data, fake_action

        mapping = {
            "EvalClient": EvalClient,
            "deserialize_data": deserialize_data,
            "fake_action": fake_action,
        }
        return mapping[name]
    if name == "StreamingEpisodeRecorder":
        from .vis_utils import StreamingEpisodeRecorder

        return StreamingEpisodeRecorder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
