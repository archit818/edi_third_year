"""
Integration tests for fault tolerance, checkpoint recovery, and priority preemption.
"""
import time
import shutil
import tempfile
import asyncio
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from distri_grid.master.checkpoint_manager import CheckpointManager
from distri_grid.master.worker_manager import WorkerManager
from distri_grid.master.data_streamer import DataStreamer
from distri_grid.master.dispatcher import Dispatcher
from distri_grid.protocol.messages import (
    WorkerRegistration,
    HeartbeatMessage,
    TaskType,
    TaskState,
)
from distri_grid.protocol.serialization import serialize_model_state


def test_master_checkpoint_recovery():
    temp_dir = tempfile.mkdtemp()
    try:
        ckpt_mgr = CheckpointManager(base_dir=temp_dir)
        model = nn.Linear(5, 2)
        
        # 1. Save step 10 checkpoint with loss 0.8
        state_1 = serialize_model_state({"model_state_dict": model.state_dict()})
        ckpt_mgr.save_checkpoint("job_alpha", epoch=0, step=10, state_dict_b64=state_1, loss=0.8)

        # 2. Save step 20 checkpoint with better loss 0.4
        state_2 = serialize_model_state({"model_state_dict": model.state_dict()})
        ckpt_mgr.save_checkpoint("job_alpha", epoch=0, step=20, state_dict_b64=state_2, loss=0.4)

        # Verify metadata
        meta = ckpt_mgr.get_job_metadata("job_alpha")
        assert meta["last_step"] == 20
        assert meta["best_loss"] == 0.4
        assert len(meta["checkpoints"]) == 2

        # 3. Test Recovery from master disk
        latest = ckpt_mgr.get_latest_checkpoint("job_alpha")
        assert latest is not None
        assert latest["step"] == 20
        assert latest["loss"] == 0.4
        assert "model_state_dict" in latest
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_worker_heartbeat_timeout():
    mgr = WorkerManager()
    reg = WorkerRegistration(
        worker_id="w_timeout_test",
        name="Node1",
        hostname="pc1",
        platform="Windows",
        cpu_cores=4,
        ram_gb=8.0,
    )
    session = await mgr.register(reg, websocket=None)

    assert session.is_alive is True

    # Simulate heartbeat
    hb = HeartbeatMessage(worker_id="w_timeout_test", cpu_percent=25.0)
    await mgr.update_heartbeat(hb)
    assert session.cpu_percent == 25.0

    # Simulate node power-off / timeout by moving last_heartbeat_time back 10 seconds
    session.last_heartbeat_time = time.time() - 10.0
    assert session.is_alive is False


@pytest.mark.asyncio
async def test_priority_preemption_mining_by_ml():
    temp_dir = tempfile.mkdtemp()
    try:
        worker_mgr = WorkerManager()
        streamer = DataStreamer()
        ckpt_mgr = CheckpointManager(base_dir=temp_dir)
        dispatcher = Dispatcher(worker_mgr, streamer, ckpt_mgr)

        # Register a mock worker
        reg = WorkerRegistration(
            worker_id="w_preempt_test",
            name="NodePreempt",
            hostname="pc2",
            platform="Windows",
            cpu_cores=8,
            ram_gb=16.0,
        )
        await worker_mgr.register(reg, websocket=None)

        # 1. Start mining task
        task_id = await dispatcher.submit_mining_job(duration_seconds=30.0)
        assert dispatcher.active_mining_task_id == task_id

        # 2. Submit ML job
        X = torch.randn(20, 5)
        y = torch.randint(0, 2, (20,))
        loader = DataLoader(TensorDataset(X, y), batch_size=10)

        ml_job = await dispatcher.submit_ml_job(
            task_id="ml_priority_job",
            model_code="import torch.nn as nn\nclass Net(nn.Module):\n    def __init__(self): super().__init__(); self.fc = nn.Linear(5, 2)\n    def forward(self, x): return self.fc(x)",
            model_class_name="Net",
            hyperparams={"lr": 0.01},
            dataloader=loader,
            epochs=1,
        )

        # Verify that active mining was stopped/preempted
        assert dispatcher.active_mining_task_id is None
        assert ml_job["state"] == TaskState.RUNNING
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
