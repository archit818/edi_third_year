"""
Serialization and deserialization utilities for PyTorch tensors and messages over WebSockets.
"""
import io
import base64
import json
from typing import Dict, Any, Union, Optional
import torch
from distri_grid.protocol.messages import (
    MessageType,
    BaseMessage,
    WorkerRegistration,
    WorkerRegistrationResponse,
    HeartbeatMessage,
    TaskAssignMessage,
    TaskStatusMessage,
    BatchRequestMessage,
    BatchDataMessage,
    StepFeedbackMessage,
    CheckpointUploadMessage,
    MiningFeedbackMessage,
    CommandMessage,
)

# Registry mapping MessageType to Pydantic models
MESSAGE_CLASS_MAP = {
    MessageType.REGISTER: WorkerRegistration,
    MessageType.REGISTER_ACK: WorkerRegistrationResponse,
    MessageType.HEARTBEAT: HeartbeatMessage,
    MessageType.TASK_ASSIGN: TaskAssignMessage,
    MessageType.TASK_STATUS: TaskStatusMessage,
    MessageType.BATCH_REQUEST: BatchRequestMessage,
    MessageType.BATCH_DATA: BatchDataMessage,
    MessageType.STEP_FEEDBACK: StepFeedbackMessage,
    MessageType.CHECKPOINT_UPLOAD: CheckpointUploadMessage,
    MessageType.MINING_FEEDBACK: MiningFeedbackMessage,
    MessageType.COMMAND: CommandMessage,
}


def serialize_tensors(tensors: Dict[str, torch.Tensor]) -> str:
    """
    Serializes a dictionary of PyTorch tensors into a base64-encoded string.
    Zero-disk in-memory buffer.
    """
    buffer = io.BytesIO()
    # Detach and move to cpu before serialization for safety
    cpu_tensors = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in tensors.items()}
    torch.save(cpu_tensors, buffer)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def deserialize_tensors(payload_b64: str, device: Union[str, torch.device] = "cpu") -> Dict[str, torch.Tensor]:
    """
    Deserializes a base64-encoded string back into a dictionary of PyTorch tensors,
    directly mapped to the target device (CPU/CUDA/MPS).
    """
    raw_bytes = base64.b64decode(payload_b64.encode("utf-8"))
    buffer = io.BytesIO(raw_bytes)
    tensors = torch.load(buffer, map_location=device, weights_only=False)
    return tensors


def serialize_model_state(state_dict: Dict[str, Any]) -> str:
    """
    Serializes a model or optimizer state_dict to base64 string.
    """
    buffer = io.BytesIO()
    cpu_state = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            cpu_state[k] = v.detach().cpu()
        elif isinstance(v, dict):
            cpu_state[k] = {
                sub_k: sub_v.detach().cpu() if isinstance(sub_v, torch.Tensor) else sub_v
                for sub_k, sub_v in v.items()
            }
        else:
            cpu_state[k] = v
    torch.save(cpu_state, buffer)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def deserialize_model_state(payload_b64: str, device: Union[str, torch.device] = "cpu") -> Dict[str, Any]:
    """
    Deserializes model/optimizer state_dict from base64 string.
    """
    raw_bytes = base64.b64decode(payload_b64.encode("utf-8"))
    buffer = io.BytesIO(raw_bytes)
    return torch.load(buffer, map_location=device, weights_only=False)


def pack_message(msg: BaseMessage) -> str:
    """
    Serializes a Pydantic message model to a JSON string.
    """
    return msg.model_dump_json()


def unpack_message(data: Union[str, bytes]) -> BaseMessage:
    """
    Parses a JSON string/bytes into the corresponding typed Pydantic message.
    """
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    
    parsed = json.loads(data)
    msg_type = MessageType(parsed.get("type"))
    model_cls = MESSAGE_CLASS_MAP.get(msg_type, BaseMessage)
    return model_cls(**parsed)
