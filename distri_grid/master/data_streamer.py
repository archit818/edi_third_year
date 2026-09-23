"""
Master-side In-Memory Batch Streaming Engine for DistriGrid.
Feeds PyTorch mini-batches dynamically to connected worker GPUs on-the-fly.
The dataset remains strictly on the master's laptop RAM/disk and is never saved to the worker's filesystem.
"""
import asyncio
from typing import Dict, Any, Optional, Iterator
import torch
from torch.utils.data import DataLoader, Dataset
from distri_grid.protocol.messages import BatchDataMessage
from distri_grid.protocol.serialization import serialize_tensors


class StreamSession:
    """
    State tracking for a single active dataset streaming job.
    """
    def __init__(self, job_id: str, dataloader: DataLoader, total_epochs: int = 1):
        self.job_id = job_id
        self.dataloader = dataloader
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.total_batches_per_epoch = len(dataloader)
        self.current_step = 0
        self.batch_id_counter = 0
        self.iterator: Optional[Iterator] = None
        self._lock = asyncio.Lock()
        self._reset_iterator()

    def _reset_iterator(self):
        self.iterator = iter(self.dataloader)
        self.current_step = 0

    async def get_next_batch(self, worker_id: str) -> Optional[BatchDataMessage]:
        async with self._lock:
            if self.current_epoch >= self.total_epochs:
                return None  # Finished all epochs

            try:
                raw_batch = next(self.iterator)
            except StopIteration:
                self.current_epoch += 1
                if self.current_epoch >= self.total_epochs:
                    return None
                self._reset_iterator()
                try:
                    raw_batch = next(self.iterator)
                except StopIteration:
                    return None

            self.current_step += 1
            self.batch_id_counter += 1

            # Format batch into dict of tensors
            if isinstance(raw_batch, (list, tuple)):
                if len(raw_batch) == 2:
                    tensor_dict = {"inputs": raw_batch[0], "targets": raw_batch[1]}
                else:
                    tensor_dict = {f"tensor_{i}": t for i, t in enumerate(raw_batch)}
            elif isinstance(raw_batch, dict):
                tensor_dict = raw_batch
            elif isinstance(raw_batch, torch.Tensor):
                tensor_dict = {"inputs": raw_batch}
            else:
                raise ValueError(f"Unsupported batch type from DataLoader: {type(raw_batch)}")

            is_last = (self.current_epoch == self.total_epochs - 1) and (self.current_step == self.total_batches_per_epoch)

            payload_b64 = serialize_tensors(tensor_dict)
            return BatchDataMessage(
                task_id=self.job_id,
                batch_id=self.batch_id_counter,
                epoch=self.current_epoch,
                step=self.current_step,
                is_last_batch=is_last,
                tensor_payload_b64=payload_b64,
            )


class DataStreamer:
    """
    Manages active streaming sessions across jobs.
    """
    def __init__(self):
        self._sessions: Dict[str, StreamSession] = {}
        self._lock = asyncio.Lock()

    async def register_session(self, job_id: str, dataloader: DataLoader, epochs: int = 1) -> StreamSession:
        async with self._lock:
            session = StreamSession(job_id=job_id, dataloader=dataloader, total_epochs=epochs)
            self._sessions[job_id] = session
            return session

    async def get_next_batch(self, job_id: str, worker_id: str) -> Optional[BatchDataMessage]:
        session = self._sessions.get(job_id)
        if not session:
            return None
        return await session.get_next_batch(worker_id)

    async def remove_session(self, job_id: str):
        async with self._lock:
            if job_id in self._sessions:
                del self._sessions[job_id]

    def get_session(self, job_id: str) -> Optional[StreamSession]:
        return self._sessions.get(job_id)
