"""
Dynamic PyTorch ML Execution Engine for DistriGrid Worker Nodes.
Executes training loops on GPU using streamed batches from RAM, computes metrics,
and streams checkpoints back to the Master node.
"""
import time
import asyncio
import logging
from typing import Dict, Any, Optional, Callable, Tuple
import torch
import torch.nn as nn
import torch.optim as optim

from distri_grid.protocol.messages import (
    TaskAssignMessage,
    StepFeedbackMessage,
    CheckpointUploadMessage,
    TaskState,
)
from distri_grid.protocol.serialization import serialize_model_state, deserialize_model_state
from distri_grid.worker.streaming_dataset import StreamingBatchQueue

logger = logging.getLogger("distri_grid.ml_runner")


class MLRunner:
    """
    Executes a PyTorch training task in memory on the worker device (CUDA/CPU).
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        self.task_assign: Optional[TaskAssignMessage] = None
        self.batch_queue: Optional[StreamingBatchQueue] = None

        self.is_running = False
        self.is_paused = False
        self.checkpoint_interval_steps = 25  # Sync checkpoint every 25 steps

    def initialize_task(self, assign_msg: TaskAssignMessage, batch_queue: StreamingBatchQueue):
        """
        Dynamically builds the model, optimizer, and loss criterion.
        """
        self.task_assign = assign_msg
        self.batch_queue = batch_queue
        hyperparams = assign_msg.hyperparams or {}

        # 1. Instantiate Model
        if assign_msg.model_code and assign_msg.model_class_name:
            local_scope: Dict[str, Any] = {"torch": torch, "nn": nn}
            exec(assign_msg.model_code, globals(), local_scope)
            model_cls = local_scope.get(assign_msg.model_class_name)
            if not model_cls:
                raise ValueError(f"Class '{assign_msg.model_class_name}' not found in provided model_code.")
            self.model = model_cls()
        else:
            # Fallback simple linear/mlp model if none specified
            in_dim = hyperparams.get("input_dim", 784)
            num_classes = hyperparams.get("num_classes", 10)
            self.model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_dim, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        try:
            self.model.to(self.device)
        except Exception as e:
            logger.warning(f"Could not place model on {self.device} ({e}). Falling back to CPU.")
            self.device = torch.device("cpu")
            self.model.to(self.device)

        # 2. Setup Optimizer
        lr = hyperparams.get("lr", 0.001)
        opt_name = hyperparams.get("optimizer", "Adam").lower()
        if opt_name == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        elif opt_name == "adamw":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # 3. Setup Loss Criterion
        loss_name = hyperparams.get("loss", "CrossEntropyLoss").lower()
        if "mse" in loss_name:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        # 4. Restore Checkpoint if resuming
        if assign_msg.initial_checkpoint_b64:
            state = deserialize_model_state(assign_msg.initial_checkpoint_b64, device=self.device)
            if "model_state_dict" in state:
                self.model.load_state_dict(state["model_state_dict"])
            if "optimizer_state_dict" in state and self.optimizer:
                try:
                    self.optimizer.load_state_dict(state["optimizer_state_dict"])
                except Exception:
                    pass
            logger.info("Restored model weights from initial master checkpoint.")

        self.checkpoint_interval_steps = hyperparams.get("checkpoint_interval_steps", 25)

    async def run_training_loop(
        self,
        worker_id: str,
        on_step_feedback: Callable[[StepFeedbackMessage], Any],
        on_checkpoint: Callable[[CheckpointUploadMessage], Any],
        request_more_batches_cb: Callable[[], Any],
    ):
        """
        Consumes batches from the in-memory queue, trains the model, and yields feedback.
        """
        if not self.model or not self.optimizer or not self.criterion or not self.batch_queue:
            raise RuntimeError("MLRunner has not been initialized.")

        self.is_running = True
        self.is_paused = False
        self.model.train()

        logger.info(f"Starting PyTorch training loop on device: {self.device}")

        step_counter = 0
        while self.is_running:
            if self.is_paused:
                await asyncio.sleep(0.5)
                continue

            # Signal master to stream batches if queue has space
            request_more_batches_cb()

            # Retrieve next batch from in-memory queue (waits up to 5 seconds)
            batch_data = await self.batch_queue.get_next_batch(timeout=5.0)
            if batch_data is None:
                if self.batch_queue.is_finished:
                    logger.info("Dataset stream completed. Ending training loop.")
                    break
                # Waiting for more batches from master over network
                continue

            tensors = batch_data["tensors"]
            epoch = batch_data["epoch"]
            step = batch_data["step"]
            is_last = batch_data["is_last"]

            # Extract inputs and targets
            inputs = tensors.get("inputs")
            targets = tensors.get("targets")

            if inputs is None or targets is None:
                continue

            try:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
            except Exception as e:
                logger.warning(f"Tensor transfer to {self.device} failed ({e}). Falling back to CPU.")
                self.device = torch.device("cpu")
                self.model.to(self.device)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

            # Training forward & backward pass
            t0 = time.time()
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            step_time = max(0.0001, time.time() - t0)

            step_counter += 1
            loss_val = float(loss.item())

            # Calculate accuracy if classification
            acc_val = None
            if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                preds = torch.argmax(outputs, dim=1)
                correct = (preds == targets).sum().item()
                total = targets.size(0)
                acc_val = round((correct / total) * 100, 2)
                throughput = total / step_time
            else:
                throughput = inputs.size(0) / step_time

            # Report step feedback to master
            feedback = StepFeedbackMessage(
                task_id=self.task_assign.task_id,
                worker_id=worker_id,
                epoch=epoch,
                step=step,
                loss=round(loss_val, 4),
                accuracy=acc_val,
                throughput_samples_sec=round(throughput, 1),
                step_time_sec=round(step_time, 4),
            )
            on_step_feedback(feedback)

            # Clean memory immediately
            del inputs, targets, outputs, loss
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            # Upload checkpoint periodically or on last batch
            if (step_counter % self.checkpoint_interval_steps == 0) or is_last:
                ckpt_payload = serialize_model_state({
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                })
                ckpt_msg = CheckpointUploadMessage(
                    task_id=self.task_assign.task_id,
                    worker_id=worker_id,
                    epoch=epoch,
                    step=step,
                    loss=loss_val,
                    accuracy=acc_val,
                    state_dict_b64=ckpt_payload,
                )
                on_checkpoint(ckpt_msg)

            if is_last:
                break

        self.is_running = False

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    def stop(self):
        self.is_running = False
        if self.batch_queue:
            self.batch_queue.mark_finished()
