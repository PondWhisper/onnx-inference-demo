import torch
from infer_pytorch import SimpleModel


def main():
    model = SimpleModel()

    torch.save(model.state_dict(), "simple_model.pth")
    print("Save state_dict to simple_model.pth")

    load_model = SimpleModel()
    load_model.load_state_dict(torch.load("simple_model.pth"))
    load_model.eval()

    x = torch.randn(1, 4)

    with torch.no_grad():
        y = load_model(x)

    print("input shape", x.shape)
    print("output shape", y.shape)
    print("output", y)


if __name__ == "__main__":
    main()
