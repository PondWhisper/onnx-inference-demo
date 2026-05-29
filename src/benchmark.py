"""Compare PyTorch and ONNX Runtime inference latency."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from infer_pytorch import build_model, demo_input


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONNX_PATH = PROJECT_ROOT / "model.onnx"


def benchmark_ms(fn, repeat: int = 1000, warmup: int = 50) -> float:
    for _ in range(warmup):
        fn()

    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    end = time.perf_counter()
    return (end - start) * 1000 / repeat


def ensure_onnx_model() -> None:
    if ONNX_PATH.exists():
        return
    subprocess.run([sys.executable, str(PROJECT_ROOT / "src" / "export_onnx.py")], check=True)


def main() -> None:
    repeat = 1000
    warmup = 50

    model = build_model()
    torch_input = demo_input()

    def run_pytorch() -> None:
        with torch.no_grad():
            model(torch_input)

    ensure_onnx_model()
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    ort_input = np.array([[0.2, 0.4, 0.6, 0.8]], dtype=np.float32)

    def run_onnxruntime() -> None:
        session.run([output_name], {input_name: ort_input})

    pytorch_latency = benchmark_ms(run_pytorch, repeat=repeat, warmup=warmup)
    onnxruntime_latency = benchmark_ms(run_onnxruntime, repeat=repeat, warmup=warmup)

    print(f"repeat: {repeat}")
    print(f"warmup: {warmup}")
    print(f"PyTorch average latency:      {pytorch_latency:.4f} ms")
    print(f"ONNX Runtime average latency: {onnxruntime_latency:.4f} ms")


if __name__ == "__main__":
    main()
