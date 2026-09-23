"""
DistriGrid SDK for easy programmatic PyTorch training across remote GPU clusters.
"""
import inspect
import asyncio
import httpx
from typing import Optional, Dict, Any, Type
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from distri_grid.master.server import dispatcher, worker_manager, checkpoint_manager
from distri_grid.protocol.serialization import serialize_model_state


class DistriTrainer:
    """
    High-level Python SDK interface to train PyTorch models on the DistriGrid cluster.
    """

    def __init__(self, master_url: str = "http://localhost:8000"):
        self.master_url = master_url.rstrip("/")

    @staticmethod
    def extract_model_source(model_class: Type[nn.Module]) -> str:
        """Extracts source code definition of a PyTorch nn.Module class."""
        return inspect.getsource(model_class)

    async def fit_async(
        self,
        model_class: Type[nn.Module],
        dataloader: DataLoader,
        epochs: int = 5,
        lr: float = 0.001,
        optimizer: str = "Adam",
        loss_fn: str = "CrossEntropyLoss",
        job_id: Optional[str] = None,
        checkpoint_interval_steps: int = 25,
    ) -> Dict[str, Any]:
        """
        Dispatches and streams the dataset to connected remote GPU worker(s).
        """
        model_code = self.extract_model_source(model_class)
        model_name = model_class.__name__
        task_id = job_id or f"train_{model_name.lower()}_{int(asyncio.get_event_loop().time())}"

        hyperparams = {
            "lr": lr,
            "optimizer": optimizer,
            "loss": loss_fn,
            "epochs": epochs,
            "checkpoint_interval_steps": checkpoint_interval_steps,
        }

        # Dispatch job via local dispatcher
        job_info = await dispatcher.submit_ml_job(
            task_id=task_id,
            model_code=model_code,
            model_class_name=model_name,
            hyperparams=hyperparams,
            dataloader=dataloader,
            epochs=epochs,
        )
        return job_info
