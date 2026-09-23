"""
Master Server for DistriGrid.
FastAPI and WebSocket application coordinating remote GPU worker nodes.
"""
import time
import asyncio
import logging
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from distri_grid.protocol.messages import (
    MessageType,
    WorkerRegistration,
    WorkerRegistrationResponse,
    HeartbeatMessage,
    BaseMessage,
    BatchRequestMessage,
    StepFeedbackMessage,
    CheckpointUploadMessage,
    MiningFeedbackMessage,
    TaskStatusMessage,
    TaskState,
)
from distri_grid.protocol.serialization import pack_message, unpack_message
from distri_grid.master.worker_manager import WorkerManager
from distri_grid.master.data_streamer import DataStreamer
from distri_grid.master.checkpoint_manager import CheckpointManager
from distri_grid.master.dispatcher import Dispatcher

logger = logging.getLogger("distri_grid.master")

app = FastAPI(title="DistriGrid Master Control Hub", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global master singleton instances
worker_manager = WorkerManager()
data_streamer = DataStreamer()
checkpoint_manager = CheckpointManager()
dispatcher = Dispatcher(worker_manager, data_streamer, checkpoint_manager)


# REST Request Models
class MineStartRequest(BaseModel):
    duration_seconds: Optional[float] = None
    difficulty: int = 4
    threads: int = 0
    algorithm: str = "sha256_custom"


@app.get("/api/status")
async def get_cluster_status():
    """Returns top-level cluster summary, active jobs, and stats."""
    cluster_stats = worker_manager.get_cluster_stats()
    active_jobs = list(dispatcher.active_jobs.values())
    return {
        "status": "online",
        "cluster": cluster_stats,
        "active_jobs": active_jobs,
        "active_mining_task": dispatcher.active_mining_task_id,
    }


@app.get("/api/workers")
async def get_workers():
    """Returns list and live telemetry of all connected worker nodes."""
    workers = [w.to_dict() for w in worker_manager.get_all_workers()]
    return {"workers": workers}


@app.get("/api/checkpoints")
async def get_checkpoints():
    """Returns all model checkpoints saved on master laptop storage."""
    return {"jobs": checkpoint_manager.list_saved_jobs()}


@app.post("/api/mine/start")
async def start_mining(req: MineStartRequest):
    """Starts on-demand compute/mining across all connected worker PCs."""
    if not worker_manager.get_online_workers():
        raise HTTPException(status_code=400, detail="No online workers connected to start compute task.")
    
    task_id = await dispatcher.submit_mining_job(
        duration_seconds=req.duration_seconds,
        difficulty=req.difficulty,
        threads=req.threads,
        algorithm=req.algorithm,
    )
    return {"status": "started", "task_id": task_id, "duration_seconds": req.duration_seconds}


@app.post("/api/mine/stop")
async def stop_mining():
    """Stops any active compute/mining jobs on all workers."""
    await dispatcher.stop_mining()
    return {"status": "stopped"}


class SyntheticJobRequest(BaseModel):
    name: str = "synthetic"
    epochs: int = 3
    batch_size: int = 32
    lr: float = 0.005


@app.post("/api/jobs/synthetic")
async def run_synthetic_job(req: SyntheticJobRequest):
    """Generates a dataset in master RAM and streams it to worker GPUs."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if not worker_manager.get_online_workers():
        raise HTTPException(status_code=400, detail="No online workers connected to accept ML job.")

    num_samples = 640
    in_features = 32
    num_classes = 4

    X = torch.randn(num_samples, in_features)
    weights = torch.randn(in_features, num_classes)
    y = torch.argmax(torch.matmul(X, weights) + 0.1 * torch.randn(num_samples, num_classes), dim=1)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=req.batch_size, shuffle=True)

    model_code = """
import torch
import torch.nn as nn

class ClassifierNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 4),
        )

    def forward(self, x):
        return self.net(x)
"""
    task_id = f"demo_ml_{int(time.time())}"
    job_info = await dispatcher.submit_ml_job(
        task_id=task_id,
        model_code=model_code,
        model_class_name="ClassifierNet",
        hyperparams={"lr": req.lr, "optimizer": "Adam", "loss": "CrossEntropyLoss"},
        dataloader=dataloader,
        epochs=req.epochs,
    )
    return {"status": "submitted", "task_id": task_id, "job": job_info}


@app.get("/api/jobs/{task_id}")
async def get_job_info(task_id: str):
    """Returns real-time status and metrics of a job."""
    job = dispatcher.active_jobs.get(task_id)
    if not job:
        # Check saved checkpoint metadata
        meta = checkpoint_manager.get_job_metadata(task_id)
        if meta and meta.get("checkpoints"):
            return {"job": meta, "status": "completed"}
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job": job}


@app.websocket("/ws/worker/{worker_id}")
async def worker_websocket_endpoint(websocket: WebSocket, worker_id: str):
    """
    Bi-directional communication channel with a remote worker PC.
    Handles zero-disk mini-batch streaming, heartbeats, and checkpoint sync.
    """
    await websocket.accept()
    logger.info(f"Worker connected: {worker_id}")

    session = None
    try:
        while True:
            raw_data = await websocket.receive_text()
            msg = unpack_message(raw_data)

            if msg.type == MessageType.REGISTER:
                reg_msg: WorkerRegistration = msg
                session = await worker_manager.register(reg_msg, websocket)
                ack = WorkerRegistrationResponse(
                    worker_id=worker_id,
                    accepted=True,
                    message=f"Registered node {reg_msg.name} successfully."
                )
                await websocket.send_text(pack_message(ack))
                logger.info(f"Worker registered: {reg_msg.name} ({reg_msg.gpu_name or 'CPU'})")

            elif msg.type == MessageType.HEARTBEAT:
                hb_msg: HeartbeatMessage = msg
                await worker_manager.update_heartbeat(hb_msg)

            elif msg.type == MessageType.BATCH_REQUEST:
                req_msg: BatchRequestMessage = msg
                batch = await data_streamer.get_next_batch(req_msg.task_id, worker_id)
                if batch:
                    await websocket.send_text(pack_message(batch))
                else:
                    # End of streaming dataset reached
                    end_batch = BatchDataMessage(
                        task_id=req_msg.task_id,
                        batch_id=-1,
                        epoch=-1,
                        step=-1,
                        is_last_batch=True,
                        tensor_payload_b64="",
                    )
                    await websocket.send_text(pack_message(end_batch))

            elif msg.type == MessageType.STEP_FEEDBACK:
                step_msg: StepFeedbackMessage = msg
                await dispatcher.handle_step_feedback(step_msg)

            elif msg.type == MessageType.CHECKPOINT_UPLOAD:
                ckpt_msg: CheckpointUploadMessage = msg
                saved_path = await dispatcher.handle_checkpoint_upload(ckpt_msg)
                ack = BaseMessage(type=MessageType.CHECKPOINT_ACK)
                await websocket.send_text(pack_message(ack))

            elif msg.type == MessageType.MINING_FEEDBACK:
                mining_msg: MiningFeedbackMessage = msg
                await dispatcher.handle_mining_feedback(mining_msg)

            elif msg.type == MessageType.TASK_STATUS:
                status_msg: TaskStatusMessage = msg
                if session:
                    session.task_state = status_msg.state
                if status_msg.state == TaskState.COMPLETED:
                    await dispatcher.complete_job(status_msg.task_id, message=status_msg.message or "Completed")

    except WebSocketDisconnect:
        logger.warning(f"Worker disconnected: {worker_id}")
    except Exception as e:
        logger.error(f"Error handling worker {worker_id}: {e}")
    finally:
        await worker_manager.disconnect(worker_id)
