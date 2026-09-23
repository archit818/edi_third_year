"""
Master-side Checkpoint Manager for DistriGrid.
Persists model and optimizer weights strictly on the master node (user's laptop)
to ensure 100% fault tolerance if remote worker PCs turn off or disconnect.
"""
import os
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch
from distri_grid.protocol.serialization import deserialize_model_state


class CheckpointManager:
    """
    Manages saving, loading, and tracking model checkpoints on the master filesystem.
    """

    def __init__(self, base_dir: str = "./checkpoints"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_job_dir(self, job_id: str) -> Path:
        job_dir = self.base_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    def save_checkpoint(
        self,
        job_id: str,
        epoch: int,
        step: int,
        state_dict_b64: str,
        loss: float,
        accuracy: Optional[float] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Saves a serialized checkpoint received from a worker to the master disk.
        """
        job_dir = self.get_job_dir(job_id)
        ckpt_filename = f"checkpoint_epoch{epoch}_step{step}.pt"
        ckpt_path = job_dir / ckpt_filename

        # Deserialize in memory to verify & save cleanly to disk
        state_data = deserialize_model_state(state_dict_b64, device="cpu")
        
        save_payload = {
            "job_id": job_id,
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "accuracy": accuracy,
            "hyperparams": hyperparams or {},
            "timestamp": time.time(),
            "model_state_dict": state_data.get("model_state_dict", state_data),
            "optimizer_state_dict": state_data.get("optimizer_state_dict", {}),
        }

        # Save specific step checkpoint
        torch.save(save_payload, ckpt_path)

        # Update latest.pt symlink/file
        latest_path = job_dir / "latest.pt"
        torch.save(save_payload, latest_path)

        # Update metadata index
        meta_path = job_dir / "metadata.json"
        meta = self.get_job_metadata(job_id)
        best_loss = meta.get("best_loss")
        is_best = best_loss is None or loss < best_loss

        if is_best:
            best_path = job_dir / "best.pt"
            torch.save(save_payload, best_path)
            meta["best_loss"] = loss
            meta["best_checkpoint"] = str(ckpt_path)

        meta["last_epoch"] = epoch
        meta["last_step"] = step
        meta["latest_loss"] = loss
        meta["latest_accuracy"] = accuracy
        meta["last_updated"] = time.time()
        
        if "checkpoints" not in meta:
            meta["checkpoints"] = []
        meta["checkpoints"].append({
            "filename": ckpt_filename,
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "accuracy": accuracy,
            "timestamp": time.time(),
        })

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return str(ckpt_path)

    def get_latest_checkpoint(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Loads the latest checkpoint for a job, if it exists.
        """
        job_dir = self.get_job_dir(job_id)
        latest_path = job_dir / "latest.pt"
        if not latest_path.exists():
            return None
        return torch.load(latest_path, map_location="cpu", weights_only=False)

    def get_job_metadata(self, job_id: str) -> Dict[str, Any]:
        """
        Retrieves the metadata log for a given job.
        """
        job_dir = self.get_job_dir(job_id)
        meta_path = job_dir / "metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"job_id": job_id, "checkpoints": []}

    def list_saved_jobs(self) -> List[Dict[str, Any]]:
        """
        Lists all jobs currently saved on disk.
        """
        jobs = []
        if not self.base_dir.exists():
            return jobs
        for job_folder in self.base_dir.iterdir():
            if job_folder.is_dir():
                meta = self.get_job_metadata(job_folder.name)
                jobs.append(meta)
        return jobs
