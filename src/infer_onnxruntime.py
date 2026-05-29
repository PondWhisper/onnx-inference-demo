"""Load model.onnx and run inference with ONNX Runtime."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONNX_PATH = PROJECT_ROOT / "model.onnx"


def main() -> None:
    if not ONNX_PATH.exists():
        raise FileNotFoundError(
            f"{ONNX_PATH} does not exist. Run `python src/export_onnx.py` first."
        )

    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    x = np.array([[0.2, 0.4, 0.6, 0.8]], dtype=np.float32)
    output = session.run([output_name], {input_name: x})[0]

    print("input name:", input_name)
    print("output name:", output_name)
    print("input shape:", x.shape)
    print("output shape:", output.shape)
    print("output value:", output)


if __name__ == "__main__":
    main()
