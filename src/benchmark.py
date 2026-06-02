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


def benchmark_pytorch(runs: int = 1000) -> float:
    model = SimpleModel()
    model.eval()

    x = torch.randn(1, 4)

    with torch.no_grad():
        for _ in range(10):
            model(x)

    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(runs):
            model(x)

    end = time.perf_counter()

    avg_lantency_ms = (end - start) / runs * 1000
    return avg_lantency_ms


def benchmark_onnx(runs: int = 1000) -> float:
    session = ort.InferenceSession("model.onnx")

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    x = np.random.randn(1, 4).astype(np.float32)

    for _ in range(10):
        session.run([output_name], {input_name: x})

    start = time.perf_counter()

    for _ in range(runs):
        session.run([output_name], {input_name: x})

    end = time.perf_counter()

    avg_latency_ms = (end - start) / runs * 1000
    return avg_latency_ms


def main():
    runs = 1000

    print(f"Runs:{runs}.")
    print(f"benchmark_pytorch:{benchmark_pytorch(runs): .6f}ms")
    print(f"benchmark_pytorch:{benchmark_onnx(runs): .6f}ms")


if __name__ == "__main__":
    main()
