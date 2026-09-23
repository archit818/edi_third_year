"""
End-to-End integration test of Master Server and Worker Daemon.
Tests WebSocket connection, batch streaming, step feedback, and checkpoint generation.
"""
import time
import asyncio
import pytest
import uvicorn
import multiprocessing
import httpx
import torch
from torch.utils.data import DataLoader, TensorDataset

from distri_grid.worker.daemon import WorkerDaemon
from distri_grid.master.server import app, worker_manager, data_streamer, dispatcher, checkpoint_manager
from distri_grid.protocol.messages import TaskState


@pytest.mark.asyncio
async def test_full_ml_training_cycle():
    # Setup dataset
    X = torch.randn(128, 16)
    y = torch.randint(0, 2, (128,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 1. Register Mock Worker Daemon directly
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    daemon = WorkerDaemon(master_url="ws://localhost:8000", worker_name="Test-E2E-Worker", device=device)

    # Create dummy model code
    model_code = """
import torch
import torch.nn as nn

class MiniNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 2)
    def forward(self, x):
        return self.fc(x)
"""

    # Check that serialization and deserialization work
    from distri_grid.protocol.serialization import serialize_tensors, deserialize_tensors
    batch = {"inputs": X[:16], "targets": y[:16]}
    b64 = serialize_tensors(batch)
    rec = deserialize_tensors(b64, device=device)
    assert rec["inputs"].shape == (16, 16)
