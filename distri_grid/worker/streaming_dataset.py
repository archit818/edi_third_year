"""
Worker-side Zero-Disk Streaming Dataset and Batch Queue.
Fetches mini-batches over WebSocket directly into RAM/VRAM for PyTorch training,
guaranteeing no dataset files are ever written to the worker PC's disk.
"""
import asyncio
from typing import Dict, Any, Optional
import torch
from distri_grid.protocol.serialization import deserialize_tensors


class StreamingBatchQueue:
    """
    Asynchronous in-memory batch queue for PyTorch workers.
    Maintains a small pre-fetch buffer in RAM to eliminate network wait latency during training.
    """

    def __init__(self, max_prefetch: int = 4, device: str = "cpu"):
        self.device = device
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_prefetch)
        self.is_finished = False

    async def put_batch(self, payload_b64: str, epoch: int, step: int, is_last: bool):
        """
        Deserializes received batch in memory and queues it for the training runner.
        """
        tensors = deserialize_tensors(payload_b64, device=self.device)
        await self.queue.put({
            "tensors": tensors,
            "epoch": epoch,
            "step": step,
            "is_last": is_last,
        })
        if is_last:
            self.is_finished = True

    async def get_next_batch(self, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        """
        Retrieves next batch for training. Returns None when dataset stream is exhausted.
        """
        if self.is_finished and self.queue.empty():
            return None

        try:
            item = await asyncio.wait_for(self.queue.get(), timeout=timeout)
            return item
        except asyncio.TimeoutError:
            return None

    def mark_finished(self):
        self.is_finished = True

    def clear(self):
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Exception:
                break
        self.is_finished = False
