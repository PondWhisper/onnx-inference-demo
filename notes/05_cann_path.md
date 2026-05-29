# 05 CANN Path

这篇笔记用来把当前 ONNX Demo 和 CANN / Ascend 岗位联系起来。

Ascend 是华为昇腾 AI 处理器，主要面向 AI 训练和推理场景。CANN 是 Compute Architecture for Neural Networks，可以理解成昇腾 AI 处理器的软件栈，负责把上层模型和底层 Ascend 硬件连接起来。它里面会涉及模型转换、算子执行、图优化、内存管理和运行时接口等内容。

ATC 是 Ascend Tensor Compiler，是 CANN 里的模型转换工具。常见链路是先从 PyTorch 导出 ONNX，然后用 ATC 把 ONNX 转成 OM 模型。OM 是 Ascend 侧可执行的离线模型格式，可以被 Ascend Runtime 加载并执行。

我现在这个项目实际跑通的是：

```text
PyTorch 模型
↓
ONNX 模型
↓
ONNX Runtime 推理
```

如果放到 Ascend / CANN 部署方向，后续链路会变成：

```text
PyTorch 模型
↓
ONNX 模型
↓
ATC 转换
↓
OM 模型
↓
Ascend / CANN 推理
```

当前项目边界很清楚：我没有 Ascend 硬件，也没有实际完成 ATC 转换和 OM 推理。因此我不能说自己做过完整 CANN 部署。但这个项目已经帮我理解了前半段链路：模型如何从 PyTorch 框架中导出，如何被 Runtime 加载执行，以及为什么模型部署需要中间表示和运行时。

面试中可以这样表达：我目前用 PyTorch、ONNX、ONNX Runtime 跑通了一个最小推理链路，正在沿着 ONNX -> ATC -> OM -> Ascend Runtime 的方向继续补 CANN 知识。我的优势不是已经精通 CANN，而是能把一个方向拆成可验证的小实验，并持续补齐部署链路。
