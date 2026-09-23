"""
Message schemas and data structures for DistriGrid network communication.
"""
from enum import Enum
from typing import Optional, Dict, Any, List
import time
from pydantic import BaseModel, Field


class MessageType(str, Enum):
    REGISTER = "register"
    REGISTER_ACK = "register_ack"
    HEARTBEAT = "heartbeat"
    HEARTBEAT_ACK = "heartbeat_ack"
    TASK_ASSIGN = "task_assign"
    TASK_STATUS = "task_status"
    BATCH_REQUEST = "batch_request"
    BATCH_DATA = "batch_data"
    STEP_FEEDBACK = "step_feedback"
    CHECKPOINT_UPLOAD = "checkpoint_upload"
    CHECKPOINT_ACK = "checkpoint_ack"
    MINING_FEEDBACK = "mining_feedback"
    COMMAND = "command"
    ERROR = "error"


class TaskType(str, Enum):
    ML_TRAIN = "ml_train"
    COMPUTE_MINE = "compute_mine"
    BENCHMARK = "benchmark"


class TaskState(str, Enum):
    IDLE = "idle"
    PENDING = "pending"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BaseMessage(BaseModel):
    type: MessageType
    timestamp: float = Field(default_factory=time.time)


class WorkerRegistration(BaseMessage):
    type: MessageType = MessageType.REGISTER
    worker_id: str
    name: str
    hostname: str
    platform: str
    cpu_cores: int
    ram_gb: float
    gpu_available: bool = False
    gpu_name: Optional[str] = None
    gpu_vram_mb: Optional[float] = None
    cuda_version: Optional[str] = None


class WorkerRegistrationResponse(BaseMessage):
    type: MessageType = MessageType.REGISTER_ACK
    worker_id: str
    accepted: bool
    cluster_name: str = "DistriGrid-Cluster"
    message: str = "Registration successful"


class HeartbeatMessage(BaseMessage):
    type: MessageType = MessageType.HEARTBEAT
    worker_id: str
    cpu_percent: float = 0.0
    ram_percent: float = 0.0
    gpu_util_percent: Optional[float] = None
    gpu_vram_used_mb: Optional[float] = None
    gpu_temp_c: Optional[float] = None
    active_task_id: Optional[str] = None
    task_state: TaskState = TaskState.IDLE


class TaskAssignMessage(BaseMessage):
    type: MessageType = MessageType.TASK_ASSIGN
    task_id: str
    task_type: TaskType
    model_code: Optional[str] = None  # Python code containing model definition
    model_class_name: Optional[str] = None
    hyperparams: Dict[str, Any] = Field(default_factory=dict)
    # E.g. {"lr": 0.001, "optimizer": "Adam", "loss": "CrossEntropyLoss", "epochs": 5, "batch_size": 64}
    duration_seconds: Optional[float] = None  # For timed compute/mining
    mining_config: Optional[Dict[str, Any]] = None
    # E.g. {"algorithm": "sha256_custom", "threads": 4, "difficulty": 4}
    initial_checkpoint_b64: Optional[str] = None  # If resuming from checkpoint


class TaskStatusMessage(BaseMessage):
    type: MessageType = MessageType.TASK_STATUS
    task_id: str
    worker_id: str
    state: TaskState
    message: Optional[str] = None
    progress_percent: Optional[float] = None


class BatchRequestMessage(BaseMessage):
    type: MessageType = MessageType.BATCH_REQUEST
    task_id: str
    worker_id: str
    request_count: int = 1
    current_epoch: int = 0
    current_step: int = 0


class BatchDataMessage(BaseMessage):
    type: MessageType = MessageType.BATCH_DATA
    task_id: str
    batch_id: int
    epoch: int
    step: int
    is_last_batch: bool = False
    tensor_payload_b64: str  # Base64 serialized tensors {inputs, targets}


class StepFeedbackMessage(BaseMessage):
    type: MessageType = MessageType.STEP_FEEDBACK
    task_id: str
    worker_id: str
    epoch: int
    step: int
    loss: float
    accuracy: Optional[float] = None
    throughput_samples_sec: float = 0.0
    step_time_sec: float = 0.0


class CheckpointUploadMessage(BaseMessage):
    type: MessageType = MessageType.CHECKPOINT_UPLOAD
    task_id: str
    worker_id: str
    epoch: int
    step: int
    loss: float
    accuracy: Optional[float] = None
    state_dict_b64: str  # Base64 serialized model & optimizer state_dict


class MiningFeedbackMessage(BaseMessage):
    type: MessageType = MessageType.MINING_FEEDBACK
    task_id: str
    worker_id: str
    hashrate_hps: float
    hashes_computed: int
    accepted_shares: int
    rejected_shares: int
    elapsed_seconds: float
    gpu_temp_c: Optional[float] = None


class CommandMessage(BaseMessage):
    type: MessageType = MessageType.COMMAND
    command: str  # "PAUSE", "RESUME", "STOP", "SWITCH_TASK"
    task_id: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)
