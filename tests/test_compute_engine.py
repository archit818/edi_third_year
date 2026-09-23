"""
Unit tests for on-demand compute and mining engine.
"""
import time
import asyncio
import pytest
from distri_grid.protocol.messages import TaskAssignMessage, TaskType, MiningFeedbackMessage
from distri_grid.worker.compute_engine import ComputeEngine


@pytest.mark.asyncio
async def test_compute_engine_execution():
    engine = ComputeEngine(default_threads=2)

    feedback_received = []

    def on_feedback(fb: MiningFeedbackMessage):
        feedback_received.append(fb)

    task_assign = TaskAssignMessage(
        task_id="test_mine_job",
        task_type=TaskType.COMPUTE_MINE,
        duration_seconds=5.0,
        mining_config={"difficulty": 2, "threads": 2},
    )

    # Start compute
    engine.start_mining(
        task_assign=task_assign,
        worker_id="test_worker_1",
        feedback_callback=on_feedback,
    )

    assert engine.is_running is True

    # Allow it to run for 3 seconds
    await asyncio.sleep(3.0)

    assert engine.total_hashes > 1000
    assert len(feedback_received) >= 1
    assert feedback_received[-1].hashes_computed > 0

    # Test Pause
    engine.pause()
    assert engine.is_paused is True
    hashes_at_pause = engine.total_hashes
    await asyncio.sleep(0.5)
    # Shouldn't increase much while paused
    assert abs(engine.total_hashes - hashes_at_pause) < 50

    # Test Stop
    engine.stop()
    assert engine.is_running is False
