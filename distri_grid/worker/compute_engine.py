"""
On-Demand Compute and Mining Engine for DistriGrid Worker Nodes.
Executes multi-threaded cryptographic hashing / benchmark compute workloads,
calculates real-time hashrate, and reports shares and telemetry to Master.
"""
import time
import hashlib
import asyncio
import threading
import logging
from typing import Dict, Any, Optional, Callable
from distri_grid.protocol.messages import MiningFeedbackMessage, TaskAssignMessage

logger = logging.getLogger("distri_grid.compute_engine")


class ComputeEngine:
    """
    High-performance multi-threaded compute worker.
    """

    def __init__(self, default_threads: int = 0):
        self.default_threads = default_threads
        self.is_running = False
        self.is_paused = False
        self.task_assign: Optional[TaskAssignMessage] = None

        self.total_hashes = 0
        self.accepted_shares = 0
        self.rejected_shares = 0
        self.start_time = 0.0
        self.current_hashrate = 0.0

        self._stop_event = threading.Event()
        self._threads = []
        self._hash_count_lock = threading.Lock()
        self._window_hashes = 0
        self._last_window_time = time.time()

    def _worker_thread(self, difficulty_prefix: str, thread_id: int):
        """Worker thread executing proof-of-work hash calculations."""
        local_nonce = thread_id * 100_000_000
        prefix_bytes = f"distrigrid_block_{thread_id}_".encode("utf-8")
        
        while not self._stop_event.is_set():
            if self.is_paused:
                time.sleep(0.1)
                continue

            local_nonce += 1
            data = prefix_bytes + str(local_nonce).encode("utf-8")
            digest = hashlib.sha256(data).hexdigest()

            with self._hash_count_lock:
                self.total_hashes += 1
                self._window_hashes += 1
                if digest.startswith(difficulty_prefix):
                    self.accepted_shares += 1

            if local_nonce % 2000 == 0:
                time.sleep(0.001)  # Yield CPU timeslice

    def start_mining(
        self,
        task_assign: TaskAssignMessage,
        worker_id: str,
        feedback_callback: Callable[[MiningFeedbackMessage], Any],
    ):
        """Starts multi-threaded compute/mining execution."""
        self.task_assign = task_assign
        cfg = task_assign.mining_config or {}
        difficulty = cfg.get("difficulty", 4)
        difficulty_prefix = "0" * difficulty
        
        num_threads = cfg.get("threads") or self.default_threads
        if not num_threads or num_threads <= 0:
            import os
            num_threads = max(1, (os.cpu_count() or 2) - 1)

        self.is_running = True
        self.is_paused = False
        self._stop_event.clear()
        self.total_hashes = 0
        self.accepted_shares = 0
        self.rejected_shares = 0
        self.start_time = time.time()
        self._last_window_time = time.time()
        self._window_hashes = 0
        self._threads = []

        logger.info(f"Starting compute/mining task {task_assign.task_id} with {num_threads} threads, difficulty: {difficulty}")

        for i in range(num_threads):
            t = threading.Thread(target=self._worker_thread, args=(difficulty_prefix, i), daemon=True)
            t.start()
            self._threads.append(t)

        # Background reporting loop
        asyncio.create_task(self._reporting_loop(worker_id, feedback_callback))

    async def _reporting_loop(
        self,
        worker_id: str,
        feedback_callback: Callable[[MiningFeedbackMessage], Any],
    ):
        while self.is_running and not self._stop_event.is_set():
            await asyncio.sleep(2.0)
            now = time.time()
            elapsed_window = max(0.001, now - self._last_window_time)

            with self._hash_count_lock:
                self.current_hashrate = self._window_hashes / elapsed_window
                self._window_hashes = 0
                self._last_window_time = now
                current_total = self.total_hashes
                current_shares = self.accepted_shares

            feedback = MiningFeedbackMessage(
                task_id=self.task_assign.task_id if self.task_assign else "mine",
                worker_id=worker_id,
                hashrate_hps=round(self.current_hashrate, 2),
                hashes_computed=current_total,
                accepted_shares=current_shares,
                rejected_shares=self.rejected_shares,
                elapsed_seconds=round(now - self.start_time, 1),
            )
            feedback_callback(feedback)

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    def stop(self):
        self.is_running = False
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=0.5)
        self._threads = []
        logger.info("Compute/Mining task stopped.")
