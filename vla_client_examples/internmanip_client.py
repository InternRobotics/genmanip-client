"""
Copied from InternManip
"""

import base64
import io
import binascii
import numpy as np
import torch
from PIL import Image
from PIL import UnidentifiedImageError
import requests
from typing import Any, Dict
from pydantic import BaseModel
from typing import Optional, Union


def encode_numpy(data: np.ndarray) -> dict:
    metadata = {
        "type": "numpy_array",
        "dtype": str(data.dtype),
        "shape": data.shape,
        "data": base64.b64encode(data.tobytes()).decode("utf-8"),
    }
    return metadata


def encode_tensor(tensor: torch.Tensor) -> dict:
    tensor_dtype = str(tensor.dtype).split(".")[-1]
    tensor_shape = str(tensor.shape)[11:-1]
    tensor_device = str(tensor.device)
    metadata = {
        "type": "tensor",
        "dtype": tensor_dtype,
        "shape": tensor_shape,
        "data": base64.b64encode(tensor.cpu().detach().numpy().tobytes()).decode(
            "utf-8"
        ),
        "device": tensor_device,
    }
    return metadata


def encode_image(image: Image.Image) -> dict:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    metadata = {
        "type": "image",
        "format": "PNG",
        "size": image.size,
        "mode": image.mode,
        "data": base64.b64encode(buffer.getvalue()).decode("utf-8"),
    }
    return metadata


def serialize_data(data):
    if isinstance(data, np.ndarray):
        return encode_numpy(data)
    elif isinstance(data, torch.Tensor):
        return encode_tensor(data)
    elif isinstance(data, Image.Image):
        return encode_image(data)
    elif isinstance(data, (list, tuple)):
        return [serialize_data(item) for item in data]
    elif isinstance(data, dict):
        return {key: serialize_data(value) for key, value in data.items()}
    else:
        return data


def decode_numpy(metadata: dict) -> np.ndarray:
    decoded_bytes = base64.b64decode(metadata["data"])
    numpy_array = np.frombuffer(decoded_bytes, dtype=np.dtype(metadata["dtype"]))
    numpy_array = numpy_array.reshape(metadata["shape"])
    return numpy_array


def decode_tensor(metadata: dict) -> torch.Tensor:
    decoded_bytes = base64.b64decode(metadata["data"])
    tensor = torch.frombuffer(
        bytearray(decoded_bytes), dtype=getattr(torch, metadata["dtype"])
    )
    tensor = tensor.reshape(eval(metadata["shape"]))
    return tensor.to(metadata["device"])


def decode_image(metadata: dict) -> Image.Image:
    try:
        decoded_bytes = base64.b64decode(metadata["data"])
        image = Image.open(io.BytesIO(decoded_bytes))

        if "size" in metadata and image.size != metadata["size"]:
            image = image.resize(metadata["size"], Image.Resampling.LANCZOS)

        if "mode" in metadata and image.mode != metadata["mode"]:
            image = image.convert(metadata["mode"])

        return image
    except (
        KeyError,
        TypeError,
        ValueError,
        binascii.Error,
        UnidentifiedImageError,
        OSError,
    ) as e:
        raise RuntimeError(f"Image decoding failed: {e}")


def deserialize_data(data):
    if isinstance(data, dict) and "type" in data:
        if data["type"] == "numpy_array":
            return decode_numpy(data)
        elif data["type"] == "tensor":
            return decode_tensor(data)
        elif data["type"] == "image":
            return decode_image(data)
    elif isinstance(data, (list, tuple)):
        return [deserialize_data(item) for item in data]
    elif isinstance(data, dict):
        return {key: deserialize_data(value) for key, value in data.items()}
    else:
        return data


class AgentCfg(BaseModel):
    agent_type: str
    model_type: str
    model_path: str
    model_kwargs: Dict[str, Any] = {}
    agent_settings: Dict[str, Any] = {}


class ServerCfg(BaseModel):
    host: str = "localhost"
    port: Optional[Union[int, str]] = "auto"


class AgentClient:
    """
    Client class for Agent service.
    """

    def __init__(self, agent_config: AgentCfg, server_config: ServerCfg):
        self.base_url = f"http://{server_config.host}:{server_config.port}"
        self.agent_name = self._initialize_agent_model(agent_config)

    def _initialize_agent_model(self, config: AgentCfg):
        request_data = config.model_dump()
        response = requests.post(
            url=f"{self.base_url}/agent/initialize",
            json=request_data,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        return response.json()["agent_name"]

    def __getattr__(self, attr_name: str):
        def attribute(*args, **kwargs):
            request_data = {
                "args": serialize_data(args),
                "kwargs": serialize_data(kwargs),
            }
            response = requests.post(
                url=f"{self.base_url}/agent/{self.agent_name}/{attr_name}",
                json=request_data,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return deserialize_data(response.json())

        return attribute
