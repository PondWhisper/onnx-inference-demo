# 01 PyTorch Inference

这一步对应 `src/infer_pytorch.py`，目标是先用 PyTorch 把一个最小模型跑起来。

Tensor 可以理解成 PyTorch 里的多维数组。模型输入、权重、输出结果本质上都以 Tensor 的形式存在。shape 描述 Tensor 的维度，例如本项目输入是 `(1, 4)`，意思是 batch size 为 1，每条样本有 4 个特征。输出是 `(1, 2)`，意思是模型对这 1 条样本输出 2 个数。

`nn.Module` 是 PyTorch 里定义模型的基础类。我们通过继承 `nn.Module` 创建 `SimpleModel`，在 `__init__` 里定义网络层，在 `forward` 里描述数据如何经过这些层。调用 `model(x)` 时，PyTorch 实际会执行 `forward(x)`。

推理和训练不同。训练时需要计算 loss、反向传播、更新参数；推理时只需要把输入送进模型，拿到输出。`model.eval()` 表示模型进入推理模式，它会关闭 Dropout、固定 BatchNorm 等训练阶段行为。`torch.no_grad()` 表示这段代码不需要记录梯度，可以减少内存和计算开销。

这一阶段我需要掌握的重点不是复杂网络结构，而是理解模型推理的基本动作：构造输入 Tensor，确认 shape，执行 forward，读取输出结果。只有先理解 PyTorch 模型如何跑起来，后面导出 ONNX、交给 Runtime 执行才有基础。


# PyTorch Inference

## 1. forward 是什么？

forward 是模型从输入到输出的计算过程。运行model的时候输入参数会自动运行这个函数。

## 2. x = torch.randn(1, 4) 里的 1 和 4 是什么？

1 是 batch size，表示一次输入 1 条样本。
4 是 feature dimension，表示每条样本有 4 个特征。

## 3. Linear(4, 8) 做了什么？

它把每条 4 维输入转换成 8 维输出。
所以 shape 从 [1, 4] 变成 [1, 8]。
提取出8个features

## 4. ReLU 做了什么？

ReLU 把负数变成 0，正数保持不变。
它不改变 tensor 的 shape。
符合大脑的工作状态：有的神经不活跃 有的活跃。

## 5. Linear(8, 2) 做了什么？

它把 8 维中间表示转换成 2 维输出。
所以 shape 从 [1, 8] 变成 [1, 2]。

## 6. 为什么输入 [1, 5] 会报错？

因为第一层 Linear(4, 8) 要求输入最后一维是 4。
如果输入是 [1, 5]，维度不匹配，所以无法计算。

## 7. model.eval() 是什么？

model.eval() 让模型进入推理模式。
model.train() 让模型进入训练模式。

## 8. torch.no_grad() 是什么？

torch.no_grad() 表示推理时不记录梯度，减少额外开销。
具有梯度的时候会把这个数据的训练过程进行反向推理，有的eval的时候不需要。

## 9. 训练和推理有什么区别？

训练会根据 loss 和梯度更新模型参数。
推理只使用已有参数执行 forward，不更新参数。

## 10. state_dict 是什么？

state_dict 是 PyTorch 模型参数的字典，里面保存了每一层可学习参数，比如 weight 和 bias。

## 11. simple_model.pth 保存了什么？

simple_model.pth 保存的是模型参数，不是完整的 Python 模型代码。模型要重新使用时，需要先创建同样结构的模型，再加载这些参数。

## 12. 为什么 ReLU 不在 state_dict 里？

ReLU 没有可学习参数，它只是对输入做一个固定操作：负数变成 0，正数保持不变。所以 state_dict 里没有 relu.weight 或 relu.bias。

## 13. 为什么模型结构不一致会加载失败？

因为保存下来的参数 shape 和当前模型结构要求的参数 shape 不匹配。比如原来 layer2 是 Linear(8,2)，现在改成 Linear(8,3)，参数维度就对不上。