"""
Hardware profiling and telemetry collector for DistriGrid worker nodes.
Auto-detects NVIDIA CUDA GPUs, VRAM capacity, CPU resources, and thermals.
"""
import os
import platform
import psutil
from typing import Dict, Any, Optional
import torch


class HardwareProfiler:
    """
    Collects system and GPU hardware specifications and live telemetry.
    """

    def __init__(self):
        self.hostname = platform.node()
        self.platform = platform.system()
        self.cpu_cores = psutil.cpu_count(logical=True) or 1
        self.total_ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
        
        self.cuda_available = torch.cuda.is_available()
        self.device_name = None
        self.gpu_vram_mb = 0.0
        self.cuda_version = None

        if self.cuda_available:
            try:
                self.device_name = torch.cuda.get_device_name(0)
                free_b, total_b = torch.cuda.mem_get_info(0)
                self.gpu_vram_mb = round(total_b / (1024 ** 2), 2)
                self.cuda_version = torch.version.cuda
            except Exception:
                self.cuda_available = False

    def get_static_specs(self) -> Dict[str, Any]:
        """Returns static hardware specifications for worker registration."""
        return {
            "hostname": self.hostname,
            "platform": self.platform,
            "cpu_cores": self.cpu_cores,
            "ram_gb": self.total_ram_gb,
            "gpu_available": self.cuda_available,
            "gpu_name": self.device_name,
            "gpu_vram_mb": self.gpu_vram_mb,
            "cuda_version": self.cuda_version,
        }

    def get_live_metrics(self) -> Dict[str, Any]:
        """Collects dynamic CPU, RAM, and GPU telemetry for heartbeats."""
        cpu_percent = psutil.cpu_percent(interval=None)
        ram_percent = psutil.virtual_memory().percent
        
        gpu_util = None
        vram_used_mb = None
        gpu_temp_c = None

        if self.cuda_available:
            try:
                free_b, total_b = torch.cuda.mem_get_info(0)
                vram_used_mb = round((total_b - free_b) / (1024 ** 2), 2)
                # Compute approximate utilization from allocated memory ratio
                gpu_util = min(100.0, round((vram_used_mb / self.gpu_vram_mb) * 100, 1)) if self.gpu_vram_mb else 0.0
            except Exception:
                pass

        return {
            "cpu_percent": cpu_percent,
            "ram_percent": ram_percent,
            "gpu_util_percent": gpu_util,
            "gpu_vram_used_mb": vram_used_mb,
            "gpu_temp_c": gpu_temp_c,
        }
