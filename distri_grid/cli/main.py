"""
Interactive CLI for DistriGrid.
Controls Master coordinator, Worker nodes, ML dataset streaming runs, and on-demand compute tasks.
"""
import sys
import time
import asyncio
from pathlib import Path
from typing import Optional
import typer
import httpx
import uvicorn
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

app = typer.Typer(help="DistriGrid: Distributed PyTorch Model Training & On-Demand Compute Grid")
console = Console()


def parse_duration(duration_str: Optional[str]) -> Optional[float]:
    """Parses duration strings like '2h', '30m', '45s', or '120' to seconds."""
    if not duration_str:
        return None
    duration_str = duration_str.strip().lower()
    if duration_str.endswith("h"):
        return float(duration_str[:-1]) * 3600
    elif duration_str.endswith("m"):
        return float(duration_str[:-1]) * 60
    elif duration_str.endswith("s"):
        return float(duration_str[:-1])
    try:
        return float(duration_str)
    except ValueError:
        raise typer.BadParameter(f"Invalid duration format: '{duration_str}'. Use e.g. '2h', '30m', '60s'.")


@app.command()
def master(
    host: str = typer.Option("0.0.0.0", help="Host interface to bind master"),
    port: int = typer.Option(8000, help="Port for master server"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
):
    """
    Starts the Master Coordinator Hub on your laptop.
    """
    console.print(Panel.fit(
        f"[bold cyan]DistriGrid Master Control Hub[/bold cyan]\n"
        f"[green]Listening on:[/green] http://{host}:{port}\n"
        f"[yellow]WebSocket endpoint:[/yellow] ws://{host}:{port}/ws/worker/{{worker_id}}\n"
        f"[dim]Run 'distri-grid worker --server http://<YOUR_IP>:{port}' on friend/lab machines to join.[/dim]",
        border_style="cyan"
    ))
    uvicorn.run("distri_grid.master.server:app", host=host, port=port, reload=reload)


@app.command()
def worker(
    server: str = typer.Option("http://localhost:8000", help="Master server URL"),
    name: Optional[str] = typer.Option(None, help="Display name for this worker node (e.g. 'Friend-Rig-1')"),
    device: Optional[str] = typer.Option(None, help="Compute device ('cuda:0' or 'cpu')"),
):
    """
    Starts the Worker node daemon on a remote desktop/GPU machine.
    """
    from distri_grid.worker.daemon import WorkerDaemon

    console.print(Panel.fit(
        f"[bold green]DistriGrid Worker Node Daemon[/bold green]\n"
        f"[cyan]Connecting to Master:[/cyan] {server}\n"
        f"[yellow]Worker Name:[/yellow] {name or 'Auto-generated'}\n"
        f"[magenta]Target Device:[/magenta] {device or 'Auto-detected CUDA/CPU'}",
        border_style="green"
    ))

    daemon = WorkerDaemon(master_url=server, worker_name=name, device=device)
    try:
        asyncio.run(daemon.start())
    except KeyboardInterrupt:
        console.print("\n[yellow]Worker daemon shutting down gracefully...[/yellow]")
        daemon.stop()


@app.command()
def status(
    server: str = typer.Option("http://localhost:8000", help="Master server URL"),
):
    """
    Inspects connected worker nodes, GPU specifications, live telemetry, and active tasks.
    """
    try:
        with httpx.Client(timeout=5.0) as client:
            resp_status = client.get(f"{server}/api/status")
            resp_workers = client.get(f"{server}/api/workers")

            if resp_status.status_code != 200 or resp_workers.status_code != 200:
                console.print(f"[bold red]Failed to get status from {server}[/bold red]")
                return

            status_data = resp_status.json()
            workers_data = resp_workers.json().get("workers", [])

            cluster = status_data.get("cluster", {})

            # Cluster Summary Banner
            console.print(Panel.fit(
                f"[bold cyan]Cluster Overview[/bold cyan]\n"
                f"[*] Online Workers: [green]{cluster.get('online_workers', 0)}[/green] / {cluster.get('total_registered', 0)}\n"
                f"[*] GPU Nodes: [magenta]{cluster.get('gpu_nodes', 0)}[/magenta]\n"
                f"[*] Total VRAM: [bold yellow]{cluster.get('total_vram_mb', 0)} MB[/bold yellow]\n"
                f"[*] Active Workloads: [bold green]{cluster.get('active_tasks', 0)}[/bold green]",
                border_style="cyan"
            ))

            # Workers Table
            table = Table(title="Connected Compute Workers", show_header=True, header_style="bold magenta")
            table.add_column("Worker ID", style="dim")
            table.add_column("Node Name", style="bold")
            table.add_column("Device / GPU", style="green")
            table.add_column("VRAM (MB)", justify="right")
            table.add_column("CPU %", justify="right")
            table.add_column("RAM %", justify="right")
            table.add_column("Task State", style="yellow")
            table.add_column("Status", style="bold")

            for w in workers_data:
                is_alive = w.get("is_alive", False)
                status_str = "[green]ONLINE[/green]" if is_alive else "[red]OFFLINE[/red]"
                gpu_str = w.get("gpu_name") or "CPU"
                vram_str = f"{w.get('gpu_vram_used_mb', 0)} / {w.get('gpu_vram_mb', 0)}" if w.get("gpu_available") else "-"
                task_str = f"{w.get('task_state', 'idle')}"
                if w.get("active_task_id"):
                    task_str += f" ({w.get('active_task_id')[:8]})"

                table.add_row(
                    str(w.get("worker_id")),
                    str(w.get("name")),
                    str(gpu_str),
                    str(vram_str),
                    f"{w.get('cpu_percent', 0)}%",
                    f"{w.get('ram_percent', 0)}%",
                    str(task_str),
                    status_str,
                )

            console.print(table)

    except Exception as e:
        console.print(f"[bold red]Could not connect to Master at {server}: {e}[/bold red]")


@app.command()
def mine(
    server: str = typer.Option("http://localhost:8000", help="Master server URL"),
    duration: Optional[str] = typer.Option("2h", help="Mining session duration (e.g. '2h', '30m', '60s')"),
    threads: int = typer.Option(0, help="Threads per worker (0 = auto)"),
    difficulty: int = typer.Option(4, help="Target proof-of-work difficulty"),
):
    """
    Dispatches on-demand compute/mining workload across all connected worker PCs.
    """
    duration_secs = parse_duration(duration)
    duration_disp = f"{duration_secs}s ({duration})" if duration_secs else "Indefinite (Manual stop)"

    console.print(Panel.fit(
        f"[bold yellow][Compute/Mining] Launching Cluster Compute Task[/bold yellow]\n"
        f"[*] Target Server: {server}\n"
        f"[*] Duration: [bold green]{duration_disp}[/bold green]\n"
        f"[*] Difficulty Prefix: {'0' * difficulty}\n"
        f"[*] Threads: {'Auto' if threads == 0 else threads}",
        border_style="yellow"
    ))

    try:
        with httpx.Client(timeout=10.0) as client:
            payload = {
                "duration_seconds": duration_secs,
                "difficulty": difficulty,
                "threads": threads,
            }
            resp = client.post(f"{server}/api/mine/start", json=payload)
            if resp.status_code == 200:
                data = resp.json()
                console.print(f"[bold green][OK] Mining task started! Task ID: {data.get('task_id')}[/bold green]")
                console.print("[dim]Use 'distri-grid status' to view live metrics or 'distri-grid stop-mining' to halt.[/dim]")
            else:
                console.print(f"[bold red]Failed to start mining: {resp.text}[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Error connecting to Master: {e}[/bold red]")


@app.command("stop-mining")
def stop_mining(
    server: str = typer.Option("http://localhost:8000", help="Master server URL"),
):
    """
    Stops all active compute/mining jobs on connected worker nodes immediately.
    """
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(f"{server}/api/mine/stop")
            if resp.status_code == 200:
                console.print("[bold green][OK] Compute/Mining jobs stopped across all worker nodes.[/bold green]")
            else:
                console.print(f"[bold red]Failed to stop mining: {resp.text}[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Error connecting to Master: {e}[/bold red]")


@app.command("run-example")
def run_example(
    name: str = typer.Option("synthetic", help="Example dataset ('synthetic' or 'mnist')"),
    epochs: int = typer.Option(3, help="Number of training epochs"),
    batch_size: int = typer.Option(32, help="Mini-batch size"),
    lr: float = typer.Option(0.005, help="Learning rate"),
    server: str = typer.Option("http://localhost:8000", help="Master server URL"),
):
    """
    Runs an end-to-end distributed ML training demonstration on the cluster with zero-disk streaming.
    """
    console.print(Panel.fit(
        f"[bold cyan][ML Training] Launching In-Memory Streaming ML Training[/bold cyan]\n"
        f"[*] Target Server: {server}\n"
        f"[*] Example: [bold]{name}[/bold]\n"
        f"[*] Epochs: {epochs}\n"
        f"[*] Batch Size: {batch_size}\n"
        f"[*] Learning Rate: {lr}\n"
        f"[dim]Dataset is created in master RAM and streamed dynamically over network to worker GPU.[/dim]",
        border_style="cyan"
    ))

    try:
        with httpx.Client(timeout=10.0) as client:
            payload = {
                "name": name,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
            }
            resp = client.post(f"{server}/api/jobs/synthetic", json=payload)
            if resp.status_code != 200:
                console.print(f"[bold red]Failed to submit ML job: {resp.text}[/bold red]")
                return

            data = resp.json()
            task_id = data.get("task_id")
            console.print(f"[bold green][OK] Job submitted to cluster: {task_id}[/bold green]")

            # Poll progress
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("| Loss: [bold red]{task.fields[loss]}[/bold red]"),
                TextColumn("| Acc: [bold green]{task.fields[acc]}[/bold green]%"),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                total_steps = (640 // batch_size) * epochs
                train_task = progress.add_task(
                    "Training ClassifierNet on Remote Worker...",
                    total=total_steps,
                    loss="--",
                    acc="--",
                )

                last_step = 0
                idle_ticks = 0
                while True:
                    time.sleep(0.5)
                    try:
                        job_resp = client.get(f"{server}/api/jobs/{task_id}")
                        if job_resp.status_code == 200:
                            resp_json = job_resp.json()
                            job_data = resp_json.get("job", {})
                            cur_step = job_data.get("current_step", 0) + (job_data.get("current_epoch", 0) * (640 // batch_size))
                            loss_val = job_data.get("latest_loss")
                            acc_val = job_data.get("latest_accuracy")
                            state = resp_json.get("status") or job_data.get("state") or job_data.get("status")

                            if cur_step > last_step:
                                advance_by = cur_step - last_step
                                last_step = cur_step
                                idle_ticks = 0
                                progress.update(
                                    train_task,
                                    advance=advance_by,
                                    loss=f"{loss_val:.4f}" if loss_val is not None else "--",
                                    acc=f"{acc_val:.1f}" if acc_val is not None else "--",
                                )
                            else:
                                idle_ticks += 1

                            # If state is marked completed, or all steps reached, or 98%+ finished with no more steps
                            if state in ("completed", TaskState.COMPLETED.value, "COMPLETED") or cur_step >= total_steps or (cur_step >= total_steps - 1 and idle_ticks >= 4):
                                progress.update(train_task, completed=total_steps)
                                break
                    except Exception:
                        pass

            console.print(f"[bold green][OK] Training completed successfully! Checkpoints saved on Master laptop.[/bold green]")
            console.print("[dim]Run 'distri-grid checkpoints' to view saved weights.[/dim]")

    except Exception as e:
        console.print(f"[bold red]Error running training job: {e}[/bold red]")


@app.command()
def checkpoints():
    """
    Lists all model checkpoints saved on the Master laptop.
    """
    from distri_grid.master.server import checkpoint_manager
    jobs = checkpoint_manager.list_saved_jobs()

    if not jobs:
        console.print("[yellow]No saved checkpoints found on this machine.[/yellow]")
        return

    table = Table(title="Saved Model Checkpoints (Master Storage)", show_header=True, header_style="bold cyan")
    table.add_column("Job ID", style="bold")
    table.add_column("Last Epoch", justify="right")
    table.add_column("Last Step", justify="right")
    table.add_column("Best Loss", justify="right", style="green")
    table.add_column("Checkpoints Count", justify="right")
    table.add_column("Best Path", style="dim")

    for j in jobs:
        table.add_row(
            j.get("job_id", "unknown"),
            str(j.get("last_epoch", "-")),
            str(j.get("last_step", "-")),
            f"{j.get('best_loss', 0):.4f}" if j.get("best_loss") else "-",
            str(len(j.get("checkpoints", []))),
            str(j.get("best_checkpoint", "-")),
        )

    console.print(table)


if __name__ == "__main__":
    app()
