import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.layer1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(8, 2)

    def forward(self, X):
        print("input:", X.shape)

        X = self.layer1(X)
        print("after layer1:", X.shape)

        X = self.relu(X)
        print("after relu:", X.shape)

        X = self.layer2(X)
        print("after layer2:", X.shape)

        return X


def main():
    model = SimpleModel()
    model.eval()

    x = torch.randn(1, 4)

    with torch.no_grad():
        y = model(x)

    print("final output:", y)


if __name__ == "__main__":
    main()
