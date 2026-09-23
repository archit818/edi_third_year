"""
Unit tests for in-memory batch streaming engine.
"""
import pytest
import asyncio
import torch
from torch.utils.data import DataLoader, TensorDataset
from distri_grid.master.data_streamer import DataStreamer
from distri_grid.worker.streaming_dataset import StreamingBatchQueue
from distri_grid.protocol.serialization import deserialize_tensors


@pytest.mark.asyncio
async def test_data_streamer_and_batch_queue():
    # 1. Create a dummy dataset with 64 samples, batch_size=16 (4 batches total per epoch)
    X = torch.randn(64, 8)
    y = torch.randint(0, 2, (64,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    streamer = DataStreamer()
    session = await streamer.register_session(job_id="test_stream_job", dataloader=dataloader, epochs=2)

    assert session.total_batches_per_epoch == 4
    assert session.total_epochs == 2

    # 2. Worker batch queue
    queue = StreamingBatchQueue(max_prefetch=4, device="cpu")

    # 3. Pull batches from streamer and put into queue
    batches_received = 0
    while True:
        batch_msg = await streamer.get_next_batch("test_stream_job", worker_id="w_test")
        if batch_msg is None:
            queue.mark_finished()
            break

        await queue.put_batch(
            payload_b64=batch_msg.tensor_payload_b64,
            epoch=batch_msg.epoch,
            step=batch_msg.step,
            is_last=batch_msg.is_last_batch,
        )
        batches_received += 1

    # Total batches over 2 epochs with 4 batches/epoch should be 8
    assert batches_received == 8

    # 4. Read from Worker StreamingQueue
    consumed_count = 0
    while True:
        item = await queue.get_next_batch(timeout=0.5)
        if item is None:
            break
        tensors = item["tensors"]
        assert "inputs" in tensors
        assert tensors["inputs"].shape[0] == 16
        consumed_count += 1

    assert consumed_count == 8
