"""
Unit tests for DistriGrid protocol messages and tensor serialization.
"""
import torch
from distri_grid.protocol.messages import (
    MessageType,
    WorkerRegistration,
    HeartbeatMessage,
    TaskAssignMessage,
    TaskType,
    TaskState,
    BatchDataMessage,
    StepFeedbackMessage,
    CheckpointUploadMessage,
)
from distri_grid.protocol.serialization import (
    serialize_tensors,
    deserialize_tensors,
    serialize_model_state,
    deserialize_model_state,
    pack_message,
    unpack_message,
)


def test_tensor_serialization():
    # Create sample tensors
    inputs = torch.randn(16, 3, 32, 32)
    targets = torch.randint(0, 10, (16,))
    batch = {"inputs": inputs, "targets": targets}

    # Serialize
    b64_str = serialize_tensors(batch)
    assert isinstance(b64_str, str)
    assert len(b64_str) > 0

    # Deserialize
    recovered = deserialize_tensors(b64_str, device="cpu")
    assert "inputs" in recovered
    assert "targets" in recovered
    assert torch.allclose(inputs, recovered["inputs"])
    assert torch.equal(targets, recovered["targets"])


def test_model_state_serialization():
    model = torch.nn.Linear(10, 2)
    state = {"model_state_dict": model.state_dict(), "step": 42}

    b64_str = serialize_model_state(state)
    assert isinstance(b64_str, str)

    recovered = deserialize_model_state(b64_str, device="cpu")
    assert "model_state_dict" in recovered
    assert recovered["step"] == 42
    assert torch.allclose(model.weight, recovered["model_state_dict"]["weight"])


def test_message_pack_unpack():
    reg = WorkerRegistration(
        worker_id="worker_test_123",
        name="TestNode",
        hostname="localhost",
        platform="Windows",
        cpu_cores=8,
        ram_gb=16.0,
        gpu_available=True,
        gpu_name="NVIDIA RTX 4050",
        gpu_vram_mb=6144.0,
    )

    packed = pack_message(reg)
    unpacked = unpack_message(packed)

    assert unpacked.type == MessageType.REGISTER
    assert unpacked.worker_id == "worker_test_123"
    assert unpacked.gpu_available is True
    assert unpacked.gpu_name == "NVIDIA RTX 4050"


def test_step_feedback_message():
    feedback = StepFeedbackMessage(
        task_id="job_001",
        worker_id="w_1",
        epoch=1,
        step=10,
        loss=0.4231,
        accuracy=94.5,
        throughput_samples_sec=320.0,
        step_time_sec=0.05,
    )

    packed = pack_message(feedback)
    unpacked = unpack_message(packed)

    assert unpacked.type == MessageType.STEP_FEEDBACK
    assert unpacked.loss == 0.4231
    assert unpacked.accuracy == 94.5
