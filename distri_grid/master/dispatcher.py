"""
Task Dispatcher for DistriGrid Master Hub.
Coordinates ML Training and Compute/Mining workloads, handles priority preemption (ML > Mining),
and manages fault recovery when worker nodes disconnect.
"""
import time
import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from distri_grid.protocol.messages import (
    TaskType,
    TaskState,
    TaskAssignMessage,
    CommandMessage,
    StepFeedbackMessage,
    CheckpointUploadMessage,
    MiningFeedbackMessage,
)
from distri_grid.protocol.serialization import pack_message
from distri_grid.master.worker_manager import WorkerManager, WorkerSession
from distri_grid.master.data_streamer import DataStreamer
from distri_grid.master.checkpoint_manager import CheckpointManager

logger = logging.getLogger("distri_grid.dispatcher")


class Dispatcher:
    """
    Central Coordinator managing workload assignments, GPU preemption, and fault tolerance.
    """

    def __init__(
        self,
        worker_manager: WorkerManager,
        data_streamer: DataStreamer,
        checkpoint_manager: CheckpointManager,
    ):
        self.worker_manager = worker_manager
        self.data_streamer = data_streamer
        self.checkpoint_manager = checkpoint_manager

        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self.active_mining_task_id: Optional[str] = None
        self._mining_timer_task: Optional[asyncio.Task] = None
        self._step_listeners: List[Callable[[StepFeedbackMessage], Any]] = []
        self._mining_listeners: List[Callable[[MiningFeedbackMessage], Any]] = []

    def add_step_listener(self, cb: Callable[[StepFeedbackMessage], Any]):
        self._step_listeners.append(cb)

    def add_mining_listener(self, cb: Callable[[MiningFeedbackMessage], Any]):
        self._mining_listeners.append(cb)

    async def submit_ml_job(
        self,
        task_id: str,
        model_code: str,
        model_class_name: str,
        hyperparams: Dict[str, Any],
        dataloader: Any,
        epochs: int = 1,
        target_worker_ids: Optional[List[str]] = None,
        initial_checkpoint_b64: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Dispatches an ML training job to worker(s). Preempts running compute/mining tasks.
        """
        # Preempt mining if active
        if self.active_mining_task_id:
            logger.info("Preempting active compute/mining job for high-priority ML training.")
            await self.stop_mining()

        # Select target workers
        available_workers = self.worker_manager.get_online_workers()
        if target_worker_ids:
            assigned_workers = [w for w in available_workers if w.worker_id in target_worker_ids]
        else:
            # Prefer GPU workers first
            gpu_workers = [w for w in available_workers if w.gpu_available]
            assigned_workers = gpu_workers if gpu_workers else available_workers

        if not assigned_workers:
            raise RuntimeError("No online workers available in cluster to accept ML training task.")

        # Register data streaming session
        await self.data_streamer.register_session(job_id=task_id, dataloader=dataloader, epochs=epochs)

        # Record active job state
        self.active_jobs[task_id] = {
            "task_id": task_id,
            "task_type": TaskType.ML_TRAIN,
            "state": TaskState.INITIALIZING,
            "workers": [w.worker_id for w in assigned_workers],
            "epochs": epochs,
            "current_epoch": 0,
            "current_step": 0,
            "latest_loss": None,
            "latest_accuracy": None,
            "model_class_name": model_class_name,
            "hyperparams": hyperparams,
            "started_at": time.time(),
            "last_step_time": time.time(),
        }

        # Build task assignment message
        assign_msg = TaskAssignMessage(
            task_id=task_id,
            task_type=TaskType.ML_TRAIN,
            model_code=model_code,
            model_class_name=model_class_name,
            hyperparams=hyperparams,
            initial_checkpoint_b64=initial_checkpoint_b64,
        )
        msg_str = pack_message(assign_msg)

        # Dispatch to assigned workers
        for worker in assigned_workers:
            if worker.websocket:
                try:
                    await worker.websocket.send_text(msg_str)
                    worker.active_task_id = task_id
                    worker.task_state = TaskState.RUNNING
                except Exception as e:
                    logger.error(f"Failed to dispatch ML task to worker {worker.worker_id}: {e}")

        self.active_jobs[task_id]["state"] = TaskState.RUNNING
        return self.active_jobs[task_id]

    async def handle_step_feedback(self, feedback: StepFeedbackMessage):
        """
        Processes step metrics returned from a worker.
        """
        job = self.active_jobs.get(feedback.task_id)
        if job:
            job["current_epoch"] = feedback.epoch
            job["current_step"] = feedback.step
            job["latest_loss"] = feedback.loss
            job["latest_accuracy"] = feedback.accuracy
            job["last_step_time"] = time.time()

        for cb in self._step_listeners:
            try:
                cb(feedback)
            except Exception:
                pass

    async def handle_checkpoint_upload(self, ckpt_msg: CheckpointUploadMessage) -> str:
        """
        Saves uploaded checkpoint safely to the master's local disk.
        """
        job = self.active_jobs.get(ckpt_msg.task_id)
        hyperparams = job.get("hyperparams", {}) if job else {}
        
        saved_path = self.checkpoint_manager.save_checkpoint(
            job_id=ckpt_msg.task_id,
            epoch=ckpt_msg.epoch,
            step=ckpt_msg.step,
            state_dict_b64=ckpt_msg.state_dict_b64,
            loss=ckpt_msg.loss,
            accuracy=ckpt_msg.accuracy,
            hyperparams=hyperparams,
        )
        logger.info(f"Saved master checkpoint for job {ckpt_msg.task_id} at epoch {ckpt_msg.epoch}, step {ckpt_msg.step} -> {saved_path}")
        return saved_path

    async def submit_mining_job(
        self,
        duration_seconds: Optional[float] = None,
        difficulty: int = 4,
        threads: int = 0,
        algorithm: str = "sha256_custom",
    ) -> str:
        """
        Broadcasts an on-demand compute/mining task to all online workers with optional duration timer.
        """
        task_id = f"mine_{int(time.time())}"
        self.active_mining_task_id = task_id

        assign_msg = TaskAssignMessage(
            task_id=task_id,
            task_type=TaskType.COMPUTE_MINE,
            duration_seconds=duration_seconds,
            mining_config={
                "algorithm": algorithm,
                "difficulty": difficulty,
                "threads": threads,
            },
        )
        msg_str = pack_message(assign_msg)

        workers = self.worker_manager.get_online_workers()
        for worker in workers:
            if worker.websocket:
                try:
                    await worker.websocket.send_text(msg_str)
                    worker.active_task_id = task_id
                    worker.task_state = TaskState.RUNNING
                except Exception as e:
                    logger.error(f"Failed to dispatch mining task to worker {worker.worker_id}: {e}")

        self.active_jobs[task_id] = {
            "task_id": task_id,
            "task_type": TaskType.COMPUTE_MINE,
            "state": TaskState.RUNNING,
            "duration_seconds": duration_seconds,
            "started_at": time.time(),
            "total_hashrate": 0.0,
            "total_shares": 0,
            "worker_hashrates": {},
        }

        # If a timed duration was set, schedule automatic stop
        if duration_seconds and duration_seconds > 0:
            if self._mining_timer_task:
                self._mining_timer_task.cancel()
            self._mining_timer_task = asyncio.create_task(self._auto_stop_mining_timer(task_id, duration_seconds))

        return task_id

    async def _auto_stop_mining_timer(self, task_id: str, duration: float):
        try:
            await asyncio.sleep(duration)
            if self.active_mining_task_id == task_id:
                logger.info(f"Compute/Mining duration of {duration}s reached. Stopping cluster compute task.")
                await self.stop_mining()
        except asyncio.CancelledError:
            pass

    async def handle_mining_feedback(self, feedback: MiningFeedbackMessage):
        """
        Updates aggregated cluster hashrates and metrics.
        """
        job = self.active_jobs.get(feedback.task_id)
        if job and job.get("task_type") == TaskType.COMPUTE_MINE:
            job["worker_hashrates"][feedback.worker_id] = {
                "hashrate_hps": feedback.hashrate_hps,
                "accepted_shares": feedback.accepted_shares,
                "gpu_temp_c": feedback.gpu_temp_c,
            }
            job["total_hashrate"] = sum(v["hashrate_hps"] for v in job["worker_hashrates"].values())
            job["total_shares"] = sum(v["accepted_shares"] for v in job["worker_hashrates"].values())

        for cb in self._mining_listeners:
            try:
                cb(feedback)
            except Exception:
                pass

    async def stop_mining(self):
        """
        Broadcasts STOP command to all workers running compute/mining.
        """
        if not self.active_mining_task_id:
            return

        cmd_msg = CommandMessage(command="STOP", task_id=self.active_mining_task_id)
        msg_str = pack_message(cmd_msg)

        for worker in self.worker_manager.get_online_workers():
            if worker.websocket:
                try:
                    await worker.websocket.send_text(msg_str)
                    if worker.active_task_id == self.active_mining_task_id:
                        worker.active_task_id = None
                        worker.task_state = TaskState.IDLE
                except Exception:
                    pass

        if self.active_mining_task_id in self.active_jobs:
            self.active_jobs[self.active_mining_task_id]["state"] = TaskState.COMPLETED

        self.active_mining_task_id = None
        if self._mining_timer_task:
            self._mining_timer_task.cancel()
            self._mining_timer_task = None

    async def complete_job(self, task_id: str, message: str = "Training finished"):
        """
        Marks an ML job as finished.
        """
        if task_id in self.active_jobs:
            self.active_jobs[task_id]["state"] = TaskState.COMPLETED
            self.active_jobs[task_id]["message"] = message

        await self.data_streamer.remove_session(task_id)

        for worker in self.worker_manager.get_online_workers():
            if worker.active_task_id == task_id:
                worker.active_task_id = None
                worker.task_state = TaskState.IDLE
