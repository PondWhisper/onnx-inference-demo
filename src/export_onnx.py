"""Export the SimpleModel from PyTorch to ONNX."""

from __future__ import annotations

from pathlib import Path

import torch

from infer_pytorch import build_model, demo_input


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONNX_PATH = PROJECT_ROOT / "model.onnx"


def main() -> None:
    model = build_model()
    x = demo_input()

    torch.onnx.export(
        model,
        x,
        ONNX_PATH,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17,
    )

    print("dummy input shape:", tuple(x.shape))
    print(f"exported ONNX model: {ONNX_PATH}")


if __name__ == "__main__":
    main()
