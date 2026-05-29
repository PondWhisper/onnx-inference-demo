"""Run one forward inference with a tiny PyTorch model."""

from __future__ import annotations

import torch
from torch import nn


class SimpleModel(nn.Module):
    """A small MLP used only to demonstrate inference and ONNX export."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def build_model() -> SimpleModel:
    torch.manual_seed(42)
    model = SimpleModel()
    model.eval()
    return model


def demo_input() -> torch.Tensor:
    return torch.tensor([[0.2, 0.4, 0.6, 0.8]], dtype=torch.float32)


def main() -> None:
    model = build_model()
    x = demo_input()

    with torch.no_grad():
        output = model(x)

    print("input shape:", tuple(x.shape))
    print("output shape:", tuple(output.shape))
    print("output value:", output.numpy())


if __name__ == "__main__":
    main()
