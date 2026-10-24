"""
Hardware profiling and telemetry collector for DistriGrid worker nodes.
Auto-detects NVIDIA CUDA GPUs, VRAM capacity, CPU resources, and thermals.
"""
import os
import platform
import subprocess
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

        # If PyTorch doesn't detect it but an NVIDIA GPU is physically present via nvidia-smi
        if not self.cuda_available:
            smi_data = self._query_nvidia_smi()
            if smi_data:
                self.gpu_vram_mb = smi_data.get("vram_total_mb", 0.0)
                try:
                    name_out = subprocess.run(
                        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                        capture_output=True, text=True, timeout=1.5
                    )
                    self.device_name = name_out.stdout.strip().split("\n")[0]
                except Exception:
                    self.device_name = "NVIDIA GPU (CUDA wheel needed)"

    def _query_nvidia_smi(self) -> Optional[Dict[str, float]]:
        """Queries nvidia-smi for accurate real-time GPU core %, temperature, and memory."""
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits"
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=1.5, check=True)
            line = res.stdout.strip().split("\n")[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                return {
                    "gpu_util_percent": float(parts[0]),
                    "gpu_temp_c": float(parts[1]),
                    "vram_used_mb": float(parts[2]),
                    "vram_total_mb": float(parts[3]),
                }
        except Exception:
            pass
        return None

    def get_static_specs(self) -> Dict[str, Any]:
        """Returns static hardware specifications for worker registration."""
        return {
            "hostname": self.hostname,
            "platform": self.platform,
            "cpu_cores": self.cpu_cores,
            "ram_gb": self.total_ram_gb,
            "gpu_available": self.cuda_available or bool(self.device_name and "NVIDIA" in self.device_name),
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

        # 1. Primary: Query nvidia-smi for accurate real-time GPU Core compute % & thermals
        smi = self._query_nvidia_smi()
        if smi:
            gpu_util = smi["gpu_util_percent"]
            gpu_temp_c = smi["gpu_temp_c"]
            vram_used_mb = smi["vram_used_mb"]
        elif self.cuda_available:
            # 2. Fallback: PyTorch CUDA info if nvidia-smi not in PATH
            try:
                free_b, total_b = torch.cuda.mem_get_info(0)
                vram_used_mb = round((total_b - free_b) / (1024 ** 2), 2)
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
