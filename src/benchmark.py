import time

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

    def forward(self, x):
        return self.net(x)


def benchmark_pytorch(batch_size: int, runs: int = 1000, warmup: int = 20) -> float:
    model = SimpleModel()
    model.eval()

    x = torch.randn(batch_size, 4)

    with torch.no_grad():
        for _ in range(warmup):
            model(x)

    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(runs):
            model(x)

    end = time.perf_counter()

    avg_latency_ms = (end - start) / runs * 1000
    return avg_latency_ms


def benchmark_onnx(batch_size: int, runs: int = 1000, warmup: int = 20) -> float:
    session = ort.InferenceSession("model.onnx")

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    x = np.random.randn(batch_size, 4).astype(np.float32)

    for _ in range(warmup):
        session.run([output_name], {input_name: x})

    start = time.perf_counter()

    for _ in range(runs):
        session.run([output_name], {input_name: x})

    end = time.perf_counter()

    avg_latency_ms = (end - start) / runs * 1000
    return avg_latency_ms


def main():
    runs = 1000
    batch_sizes = [1, 8, 32, 128]

    print(f"Runs per batch size: {runs}")
    print("| Batch size | PyTorch avg latency (ms) | ONNX Runtime CPU avg latency (ms) |")
    print("| ---: | ---: | ---: |")
    for batch_size in batch_sizes:
        pytorch_latency = benchmark_pytorch(batch_size, runs)
        onnx_latency = benchmark_onnx(batch_size, runs)
        print(f"| {batch_size} | {pytorch_latency:.6f} | {onnx_latency:.6f} |")


if __name__ == "__main__":
    main()
