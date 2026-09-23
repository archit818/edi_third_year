"""
Worker Daemon for DistriGrid.
Main client connecting to Master Hub, reporting hardware metrics, and executing workloads.
"""
import uuid
import time
import asyncio
import logging
from typing import Optional
import websockets

from distri_grid.protocol.messages import (
    MessageType,
    WorkerRegistration,
    HeartbeatMessage,
    TaskAssignMessage,
    TaskType,
    TaskState,
    BatchRequestMessage,
    BatchDataMessage,
    StepFeedbackMessage,
    CheckpointUploadMessage,
    MiningFeedbackMessage,
    CommandMessage,
    TaskStatusMessage,
)
from distri_grid.protocol.serialization import pack_message, unpack_message
from distri_grid.worker.hardware import HardwareProfiler
from distri_grid.worker.streaming_dataset import StreamingBatchQueue
from distri_grid.worker.ml_runner import MLRunner
from distri_grid.worker.compute_engine import ComputeEngine

logger = logging.getLogger("distri_grid.worker")


class WorkerDaemon:
    """
    Client daemon that connects to the Master node over WebSockets.
    """

    def __init__(
        self,
        master_url: str = "ws://localhost:8000",
        worker_name: Optional[str] = None,
        worker_id: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.master_url = master_url.rstrip("/")
        # Convert http(s) to ws(s) if provided as http
        if self.master_url.startswith("http://"):
            self.master_url = "ws://" + self.master_url[7:]
        elif self.master_url.startswith("https://"):
            self.master_url = "wss://" + self.master_url[8:]
        elif not self.master_url.startswith("ws://") and not self.master_url.startswith("wss://"):
            self.master_url = "ws://" + self.master_url

        self.worker_id = worker_id or f"worker_{uuid.uuid4().hex[:8]}"
        self.worker_name = worker_name or f"Node-{self.worker_id[-4:]}"
        
        self.profiler = HardwareProfiler()
        if device:
            self.device = device
        else:
            self.device = "cuda:0" if self.profiler.cuda_available else "cpu"

        self.ml_runner = MLRunner(device=self.device)
        self.compute_engine = ComputeEngine()
        self.batch_queue: Optional[StreamingBatchQueue] = None

        self.current_task_id: Optional[str] = None
        self.current_task_state = TaskState.IDLE
        self.is_running = False
        self.websocket = None
        self._ml_task_coro = None

    async def start(self):
        """Runs the worker client with automatic reconnect loops."""
        self.is_running = True
        endpoint = f"{self.master_url}/ws/worker/{self.worker_id}"

        logger.info(f"Connecting to Master at {endpoint}...")
        while self.is_running:
            try:
                async with websockets.connect(endpoint, max_size=100 * 1024 * 1024) as ws:
                    self.websocket = ws
                    logger.info(f"Connected to Master. Registering as '{self.worker_name}' on device '{self.device}'...")
                    
                    # 1. Send Registration
                    specs = self.profiler.get_static_specs()
                    reg = WorkerRegistration(
                        worker_id=self.worker_id,
                        name=self.worker_name,
                        hostname=specs["hostname"],
                        platform=specs["platform"],
                        cpu_cores=specs["cpu_cores"],
                        ram_gb=specs["ram_gb"],
                        gpu_available=specs["gpu_available"],
                        gpu_name=specs["gpu_name"],
                        gpu_vram_mb=specs["gpu_vram_mb"],
                        cuda_version=specs["cuda_version"],
                    )
                    await ws.send(pack_message(reg))

                    # 2. Spawn concurrent heartbeat sender and receiver loop
                    heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                    receiver_task = asyncio.create_task(self._message_receiver_loop())

                    done, pending = await asyncio.wait(
                        [heartbeat_task, receiver_task],
                        return_when=asyncio.FIRST_EXCEPTION,
                    )

                    for task in pending:
                        task.cancel()

            except (websockets.exceptions.ConnectionClosed, ConnectionRefusedError, OSError) as e:
                logger.warning(f"Connection lost to Master: {e}. Retrying in 3s...")
                await asyncio.sleep(3.0)
            except Exception as e:
                logger.error(f"Unexpected worker error: {e}. Retrying in 3s...")
                await asyncio.sleep(3.0)

    async def _heartbeat_loop(self):
        """Sends periodic heartbeats with live CPU/GPU metrics to Master."""
        while self.is_running and self.websocket:
            metrics = self.profiler.get_live_metrics()
            hb = HeartbeatMessage(
                worker_id=self.worker_id,
                cpu_percent=metrics["cpu_percent"],
                ram_percent=metrics["ram_percent"],
                gpu_util_percent=metrics["gpu_util_percent"],
                gpu_vram_used_mb=metrics["gpu_vram_used_mb"],
                gpu_temp_c=metrics["gpu_temp_c"],
                active_task_id=self.current_task_id,
                task_state=self.current_task_state,
            )
            try:
                await self.websocket.send(pack_message(hb))
            except Exception:
                break
            await asyncio.sleep(2.5)

    async def _message_receiver_loop(self):
        """Receives tasks, batches, and commands from Master."""
        while self.is_running and self.websocket:
            try:
                raw_msg = await self.websocket.recv()
                if not raw_msg:
                    continue
                
                msg = unpack_message(raw_msg)

                if msg.type == MessageType.TASK_ASSIGN:
                    task_msg: TaskAssignMessage = msg
                    await self._handle_task_assign(task_msg)

                elif msg.type == MessageType.BATCH_DATA:
                    batch_msg: BatchDataMessage = msg
                    if self.batch_queue:
                        if batch_msg.is_last_batch and not batch_msg.tensor_payload_b64:
                            self.batch_queue.mark_finished()
                        else:
                            await self.batch_queue.put_batch(
                                payload_b64=batch_msg.tensor_payload_b64,
                                epoch=batch_msg.epoch,
                                step=batch_msg.step,
                                is_last=batch_msg.is_last_batch,
                            )

                elif msg.type == MessageType.COMMAND:
                    cmd_msg: CommandMessage = msg
                    await self._handle_command(cmd_msg)
            except Exception:
                break

    async def _handle_task_assign(self, assign_msg: TaskAssignMessage):
        """Dispatches workload to MLRunner or ComputeEngine."""
        self.current_task_id = assign_msg.task_id
        self.current_task_state = TaskState.RUNNING

        if assign_msg.task_type == TaskType.ML_TRAIN:
            logger.info(f"Received ML Training task: {assign_msg.task_id}")
            # Zero-disk in-memory batch queue
            self.batch_queue = StreamingBatchQueue(max_prefetch=4, device=self.device)
            self.ml_runner.initialize_task(assign_msg, self.batch_queue)

            # Define callbacks for runner
            def on_step(feedback: StepFeedbackMessage):
                asyncio.create_task(self._send_message(feedback))

            def on_checkpoint(ckpt: CheckpointUploadMessage):
                asyncio.create_task(self._send_message(ckpt))

            def request_more_batches():
                req = BatchRequestMessage(
                    task_id=assign_msg.task_id,
                    worker_id=self.worker_id,
                    request_count=2,
                )
                asyncio.create_task(self._send_message(req))

            async def _run_ml_and_notify():
                try:
                    await self.ml_runner.run_training_loop(
                        worker_id=self.worker_id,
                        on_step_feedback=on_step,
                        on_checkpoint=on_checkpoint,
                        request_more_batches_cb=request_more_batches,
                    )
                finally:
                    self.current_task_id = None
                    self.current_task_state = TaskState.IDLE
                    status_msg = TaskStatusMessage(
                        task_id=assign_msg.task_id,
                        worker_id=self.worker_id,
                        state=TaskState.COMPLETED,
                        message="Training completed successfully",
                    )
                    await self._send_message(status_msg)

            # Run training loop in background
            self._ml_task_coro = asyncio.create_task(_run_ml_and_notify())

        elif assign_msg.task_type == TaskType.COMPUTE_MINE:
            logger.info(f"Received Compute/Mining task: {assign_msg.task_id}")
            
            def on_mining_feedback(feedback: MiningFeedbackMessage):
                asyncio.create_task(self._send_message(feedback))

            self.compute_engine.start_mining(
                task_assign=assign_msg,
                worker_id=self.worker_id,
                feedback_callback=on_mining_feedback,
            )

    async def _handle_command(self, cmd_msg: CommandMessage):
        """Processes control signals (PAUSE, RESUME, STOP)."""
        cmd = cmd_msg.command.upper()
        logger.info(f"Received Master command: {cmd}")

        if cmd == "PAUSE":
            self.ml_runner.pause()
            self.compute_engine.pause()
            self.current_task_state = TaskState.PAUSED

        elif cmd == "RESUME":
            self.ml_runner.resume()
            self.compute_engine.resume()
            self.current_task_state = TaskState.RUNNING

        elif cmd == "STOP":
            self.ml_runner.stop()
            self.compute_engine.stop()
            self.current_task_id = None
            self.current_task_state = TaskState.IDLE

    async def _send_message(self, msg):
        if self.websocket:
            try:
                await self.websocket.send(pack_message(msg))
            except Exception:
                pass

    def stop(self):
        self.is_running = False
        self.ml_runner.stop()
        self.compute_engine.stop()
