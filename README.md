# ⚡ DistriGrid

> **Distributed PyTorch Model Training & On-Demand Compute Grid**  
> Pool desktop GPUs, college lab machines, and friends' rigs seamlessly with zero-disk dataset streaming, persistent master checkpointing, and on-demand compute tasks.

---

## 🌟 Key Features

1. 🧠 **Zero-Disk In-Memory PyTorch Streaming**:
   - The dataset remains securely on your master laptop.
   - Mini-batches are serialized and streamed dynamically over WebSockets directly into remote worker RAM/VRAM, executed, and discarded immediately.
   - **Zero dataset files are ever written to remote worker disks.**

2. 🛡️ **Master-Side Persistent Checkpoints & Fault Recovery**:
   - Model weights, optimizer states, and epoch metrics are continuously synced and saved strictly onto your master laptop (`./checkpoints/<job_id>/...`).
   - If a friend's PC turns off, disconnects, or crashes, training state is never lost and can resume instantaneously.

3. ⚡ **On-Demand Distributed Compute & Crypto Pooling**:
   - Manually trigger cluster-wide multi-threaded compute/mining tasks with simple duration flags (e.g. `distri-grid mine --duration 2h`).
   - Live cluster hashrates, accepted shares, and temperature telemetry.

4. 🎛️ **Priority Task Preemption**:
   - When an ML training job is dispatched, background compute/mining tasks are automatically paused to dedicate 100% GPU power to training, and resumed when training completes.

5. 🖥️ **Rich Interactive CLI & Python SDK**:
   - Live terminal dashboards, worker telemetry tables, progress bars, loss metrics, and a simple 3-line PyTorch SDK.

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install dependencies in your virtual environment:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
```

---

### 2. Start the Master Hub (On Your Laptop)
Start the coordinator server:

```bash
distri-grid master --host 0.0.0.0 --port 8000
```

---

### 3. Connect Worker Nodes (Friends' Custom PC Rigs / College Desktops)
On each remote PC, run:

```bash
distri-grid worker --server http://<MASTER_IP>:8000 --name "Friend-Rig-RTX4060"
```
*Auto-detects NVIDIA CUDA GPUs, VRAM capacity, CPU cores, and system memory.*

---

### 4. Cluster Management Commands

#### 📊 View Cluster Status & GPU Metrics
```bash
distri-grid status --server http://localhost:8000
```

#### 🧠 Run In-Memory Distributed ML Training
```bash
distri-grid run-example --server http://localhost:8000 --epochs 3 --batch-size 32
```

#### ⚡ Launch On-Demand Compute / Mining Session (e.g. 2 hours)
```bash
distri-grid mine --server http://localhost:8000 --duration 2h --difficulty 4
```

#### 🛑 Stop Mining Across Cluster
```bash
distri-grid stop-mining --server http://localhost:8000
```

#### 💾 Inspect Saved Checkpoints on Master Laptop
```bash
distri-grid checkpoints
```

---

## 🐍 Python SDK Example

You can train any custom PyTorch `nn.Module` using the `DistriTrainer` SDK:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from distri_grid.sdk import DistriTrainer

# 1. Define Model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, 3)
        self.fc = nn.Linear(32 * 26 * 26, 10)

    def forward(self, x):
        return self.fc(torch.relu(self.conv(x)).flatten(1))

# 2. Prepare DataLoader locally in Master RAM
images = torch.randn(1000, 1, 28, 28)
labels = torch.randint(0, 10, (1000,))
dataloader = DataLoader(TensorDataset(images, labels), batch_size=32)

# 3. Train on remote cluster with zero-disk streaming
import asyncio
trainer = DistriTrainer(master_url="http://localhost:8000")
job_info = asyncio.run(trainer.fit_async(
    model_class=CNN,
    dataloader=dataloader,
    epochs=5,
    lr=0.001,
))
print("Job submitted:", job_info)
```

---

## 📁 Project Structure

```
g:/edi_third_year/
├── distri_grid/
│   ├── protocol/              # Message schemas & fast tensor serialization
│   │   ├── messages.py
│   │   └── serialization.py
│   ├── master/                # Master Hub: stream engine, checkpointing, dispatcher
│   │   ├── server.py
│   │   ├── worker_manager.py
│   │   ├── data_streamer.py
│   │   ├── checkpoint_manager.py
│   │   └── dispatcher.py
│   ├── worker/                # Worker Daemon: hardware profiler, zero-disk queue, runner
│   │   ├── daemon.py
│   │   ├── hardware.py
│   │   ├── streaming_dataset.py
│   │   ├── ml_runner.py
│   │   └── compute_engine.py
│   ├── cli/                   # Rich Terminal Interface
│   │   └── main.py
│   └── sdk/                   # High-level Python SDK
│       └── trainer.py
├── examples/                  # MNIST & Compute task scripts
├── tests/                     # Automated test suites (10/10 passed)
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 🧪 Testing

Run all unit, integration, and fault-tolerance tests:

```bash
pytest -v
```
