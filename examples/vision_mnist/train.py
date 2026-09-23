"""
Example training script demonstrating DistriTrainer SDK.
Trains MNIST model on remote worker node with in-memory batch streaming.
"""
import torch
from torch.utils.data import DataLoader, TensorDataset
from distri_grid.sdk import DistriTrainer
from examples.vision_mnist.model import MNISTConvNet


def main():
    print("=== DistriGrid MNIST Zero-Disk Distributed Training ===")
    
    # 1. Prepare sample dataset on master laptop (in RAM)
    # Using 1000 sample synthetic MNIST-like tensors for quick test
    images = torch.randn(1000, 1, 28, 28)
    labels = torch.randint(0, 10, (1000,))
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 2. Initialize DistriTrainer
    trainer = DistriTrainer(master_url="http://localhost:8000")

    # 3. Fit model on remote worker GPU
    import asyncio
    job_info = asyncio.run(trainer.fit_async(
        model_class=MNISTConvNet,
        dataloader=dataloader,
        epochs=3,
        lr=0.001,
        optimizer="Adam",
    ))

    print("Job submitted to cluster:", job_info)


if __name__ == "__main__":
    main()
