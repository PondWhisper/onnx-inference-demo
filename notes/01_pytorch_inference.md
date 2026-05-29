# 01 PyTorch Inference

这一步对应 `src/infer_pytorch.py`，目标是先用 PyTorch 把一个最小模型跑起来。

Tensor 可以理解成 PyTorch 里的多维数组。模型输入、权重、输出结果本质上都以 Tensor 的形式存在。shape 描述 Tensor 的维度，例如本项目输入是 `(1, 4)`，意思是 batch size 为 1，每条样本有 4 个特征。输出是 `(1, 2)`，意思是模型对这 1 条样本输出 2 个数。

`nn.Module` 是 PyTorch 里定义模型的基础类。我们通过继承 `nn.Module` 创建 `SimpleModel`，在 `__init__` 里定义网络层，在 `forward` 里描述数据如何经过这些层。调用 `model(x)` 时，PyTorch 实际会执行 `forward(x)`。

推理和训练不同。训练时需要计算 loss、反向传播、更新参数；推理时只需要把输入送进模型，拿到输出。`model.eval()` 表示模型进入推理模式，它会关闭 Dropout、固定 BatchNorm 等训练阶段行为。`torch.no_grad()` 表示这段代码不需要记录梯度，可以减少内存和计算开销。

这一阶段我需要掌握的重点不是复杂网络结构，而是理解模型推理的基本动作：构造输入 Tensor，确认 shape，执行 forward，读取输出结果。只有先理解 PyTorch 模型如何跑起来，后面导出 ONNX、交给 Runtime 执行才有基础。
