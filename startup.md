# 🚀 DistriGrid Startup & Quick-Start Guide

This guide explains how to start, connect, and manage **DistriGrid** for distributed PyTorch training and on-demand compute/mining across your laptop and your friends' desktop PCs.

---

## 📋 Prerequisites

Before starting, ensure the virtual environment is activated in your terminal:

```powershell
# Open PowerShell in G:\edi_third_year
.\.venv\Scripts\Activate.ps1
```

> **Note**: If PowerShell displays an Execution Policy error on Windows, run this first:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
> .\.venv\Scripts\Activate.ps1
> ```

---

## 🖥️ 1. Start the Master Hub (On Your Laptop)

The Master Server coordinates all worker nodes, streams dataset mini-batches in memory, and persists model checkpoints safely to your local disk.

```powershell
distri-grid master --host 0.0.0.0 --port 8000
```

* Keep this terminal window open.
* If Windows Firewall prompts for network access, click **Allow access**.

---

## 🔌 2. Connect Worker Nodes (Friends' PCs / Lab Machines)

On each computer joining the compute cluster:

### Step A: Clone / Copy Project Folder & Install Dependencies
Copy the repository to the worker PC and activate the environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# Automatically installs CUDA PyTorch (with CPU fallback) and distri-grid CLI:
pip install -r requirements.txt
```

*Verify GPU is detected on the friend's PC:*
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU-Only')"
```

### Step B: Find Master Laptop IP Address (On Master Laptop)
Run `ipconfig` on your laptop to find your local IPv4 address (e.g., `192.168.1.15`).

### Step C: Join Cluster as Worker
Run this on the worker PC:

```powershell
# If testing locally on the same laptop:
distri-grid worker --server http://localhost:8000 --name "My-Local-GPU"

# If connecting from a friend's PC on the same Wi-Fi / LAN:
distri-grid worker --server http://<MASTER_IP>:8000 --name "Friend-RTX3060"
```

*The worker daemon auto-detects NVIDIA CUDA GPUs, VRAM capacity, CPU cores, and system memory.*

---

## 📊 3. Inspect Cluster Telemetry

Open a new PowerShell terminal on your master laptop and run:

```powershell
distri-grid status
```

**Output displays**:
* Total registered and online workers
* Total cluster VRAM (MB)
* Live CPU %, RAM %, and GPU telemetry
* Current task state (`idle`, `running`, `paused`)

---

## 🧠 4. Run Distributed PyTorch ML Training (Zero-Disk Streaming)

Streams mini-batches dynamically from master RAM over WebSockets directly to worker GPUs. No dataset files are saved on worker disks.

```powershell
distri-grid run-example --epochs 3 --batch-size 32 --lr 0.005
```

* Features live progress bars with step loss, accuracy, and samples/sec throughput.
* Model checkpoints and optimizer states are saved to `./checkpoints/` on your master laptop.

---

## ⚡ 5. Launch On-Demand Compute / Mining Sessions

Dispatch multi-threaded compute/mining workloads across all connected PCs with duration timers:

```powershell
# 1. Run a 2-hour compute session (automatically stops after 2 hours):
distri-grid mine --duration 2h

# 2. Run a custom duration (e.g., 30 minutes or 60 seconds):
distri-grid mine --duration 30m
distri-grid mine --duration 60s

# 3. Stop mining immediately across all cluster nodes:
distri-grid stop-mining
```

> **Priority Preemption**: If you trigger an ML training job while mining is active, the Master automatically suspends the compute task to give 100% of the GPU power to model training.

---

## 💾 6. Inspect Saved Model Checkpoints

View all model checkpoints saved on your master laptop:

```powershell
distri-grid checkpoints
```

---

## 🌐 Connecting Across Different Networks (Internet)

If you and your friend are on different Wi-Fi networks:

1. **Via Ngrok**:
   ```powershell
   # On Master Laptop:
   ngrok http 8000
   
   # On Friend's PC:
   distri-grid worker --server https://<your-subdomain>.ngrok-free.app --name "Remote-Rig"
   ```

2. **Via Tailscale (Recommended)**:
   * Install [Tailscale](https://tailscale.com) on both machines.
   * Friend connects directly to your Tailscale IP:
     ```powershell
     distri-grid worker --server http://100.x.x.x:8000 --name "Friend-Rig"
     ```

---

## 🛠️ Summary of All CLI Commands

| Command | Description |
| :--- | :--- |
| `distri-grid master --port 8000` | Starts coordinator server on master laptop |
| `distri-grid worker --server <URL> --name <NAME>` | Connects GPU/CPU worker node to cluster |
| `distri-grid status` | Displays cluster summary and worker GPU tables |
| `distri-grid run-example --epochs <N>` | Runs zero-disk streaming PyTorch training demo |
| `distri-grid mine --duration <2h\|30m>` | Starts timed cluster compute/mining workload |
| `distri-grid stop-mining` | Halts compute/mining across all worker nodes |
| `distri-grid checkpoints` | Lists saved model weights on master laptop |
