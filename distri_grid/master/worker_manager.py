"""
Worker Session and Cluster Management for DistriGrid Master Node.
Tracks connected GPU/CPU nodes, monitors live heartbeats, and detects node dropouts.
"""
import time
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from fastapi import WebSocket
from distri_grid.protocol.messages import WorkerRegistration, HeartbeatMessage, TaskState


@dataclass
class WorkerSession:
    worker_id: str
    name: str
    hostname: str
    platform: str
    cpu_cores: int
    ram_gb: float
    gpu_available: bool = False
    gpu_name: Optional[str] = None
    gpu_vram_mb: Optional[float] = None
    cuda_version: Optional[str] = None
    websocket: Optional[WebSocket] = None
    registered_at: float = field(default_factory=time.time)
    last_heartbeat_time: float = field(default_factory=time.time)
    cpu_percent: float = 0.0
    ram_percent: float = 0.0
    gpu_util_percent: Optional[float] = None
    gpu_vram_used_mb: Optional[float] = None
    gpu_temp_c: Optional[float] = None
    active_task_id: Optional[str] = None
    task_state: TaskState = TaskState.IDLE

    @property
    def is_alive(self) -> bool:
        # If no heartbeat received in last 8 seconds, consider disconnected
        return (time.time() - self.last_heartbeat_time) < 8.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "name": self.name,
            "hostname": self.hostname,
            "platform": self.platform,
            "cpu_cores": self.cpu_cores,
            "ram_gb": round(self.ram_gb, 1),
            "gpu_available": self.gpu_available,
            "gpu_name": self.gpu_name or "None",
            "gpu_vram_mb": round(self.gpu_vram_mb, 1) if self.gpu_vram_mb else 0,
            "is_alive": self.is_alive,
            "cpu_percent": round(self.cpu_percent, 1),
            "ram_percent": round(self.ram_percent, 1),
            "gpu_util_percent": round(self.gpu_util_percent, 1) if self.gpu_util_percent is not None else None,
            "gpu_vram_used_mb": round(self.gpu_vram_used_mb, 1) if self.gpu_vram_used_mb is not None else None,
            "gpu_temp_c": round(self.gpu_temp_c, 1) if self.gpu_temp_c is not None else None,
            "active_task_id": self.active_task_id,
            "task_state": self.task_state.value if isinstance(self.task_state, TaskState) else self.task_state,
            "last_heartbeat_seconds_ago": round(time.time() - self.last_heartbeat_time, 1),
        }


class WorkerManager:
    """
    Registry for managing all connected compute workers.
    """

    def __init__(self):
        self._workers: Dict[str, WorkerSession] = {}
        self._lock = asyncio.Lock()

    async def register(self, reg: WorkerRegistration, websocket: WebSocket) -> WorkerSession:
        async with self._lock:
            session = WorkerSession(
                worker_id=reg.worker_id,
                name=reg.name,
                hostname=reg.hostname,
                platform=reg.platform,
                cpu_cores=reg.cpu_cores,
                ram_gb=reg.ram_gb,
                gpu_available=reg.gpu_available,
                gpu_name=reg.gpu_name,
                gpu_vram_mb=reg.gpu_vram_mb,
                cuda_version=reg.cuda_version,
                websocket=websocket,
                registered_at=time.time(),
                last_heartbeat_time=time.time(),
            )
            self._workers[reg.worker_id] = session
            return session

    async def update_heartbeat(self, hb: HeartbeatMessage):
        async with self._lock:
            worker = self._workers.get(hb.worker_id)
            if worker:
                worker.last_heartbeat_time = time.time()
                worker.cpu_percent = hb.cpu_percent
                worker.ram_percent = hb.ram_percent
                worker.gpu_util_percent = hb.gpu_util_percent
                worker.gpu_vram_used_mb = hb.gpu_vram_used_mb
                worker.gpu_temp_c = hb.gpu_temp_c
                worker.active_task_id = hb.active_task_id
                worker.task_state = hb.task_state

    async def disconnect(self, worker_id: str):
        async with self._lock:
            if worker_id in self._workers:
                del self._workers[worker_id]

    def get_worker(self, worker_id: str) -> Optional[WorkerSession]:
        return self._workers.get(worker_id)

    def get_all_workers(self) -> List[WorkerSession]:
        return list(self._workers.values())

    def get_online_workers(self) -> List[WorkerSession]:
        return [w for w in self._workers.values() if w.is_alive]

    def get_cluster_stats(self) -> Dict[str, Any]:
        all_w = list(self._workers.values())
        online_w = [w for w in all_w if w.is_alive]
        gpu_workers = [w for w in online_w if w.gpu_available]
        total_vram = sum(w.gpu_vram_mb or 0 for w in gpu_workers)
        
        return {
            "total_registered": len(all_w),
            "online_workers": len(online_w),
            "gpu_nodes": len(gpu_workers),
            "total_vram_mb": round(total_vram, 1),
            "active_tasks": len([w for w in online_w if w.active_task_id is not None and w.task_state == TaskState.RUNNING]),
        }
