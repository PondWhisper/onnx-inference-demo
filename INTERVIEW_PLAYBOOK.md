# AI Infra / CANN Interview Playbook

## 30 秒自我介绍

```text
面试官您好，我叫李敬轩，是南京邮电大学计算机科学与技术专业本科生，目前大二在读。我比较关注 AI Infra、模型推理部署和系统基础。

我最近做了一个 PyTorch -> ONNX -> ONNX Runtime 的最小推理链路 Demo，用来理解模型从训练框架走向推理 Runtime 的过程，也进一步学习它和 CANN / Ascend 中 ATC、OM、Runtime 这些概念的关系。

我目前比较熟悉 Python、Java、Linux 和 Git，也在补 C++ 和系统基础。对于 CANN 方向，我不会把自己包装成已经精通底层开发的人，而是希望从文档、工具链、小 Demo 和工程任务开始逐步深入。
```

## 1 分钟项目复述

```text
这个项目的目标不是训练复杂模型，而是跑通一个最小推理部署闭环：我先用 PyTorch 定义了一个 SimpleModel，结构是 Linear(4,8) -> ReLU -> Linear(8,2)，用 model.eval() 和 torch.no_grad() 完成推理；然后通过 torch.onnx.export 把模型导出成 ONNX 文件；再用 ONNX Runtime 的 InferenceSession 加载 model.onnx，用 numpy.float32 构造输入，通过 session.run 执行推理。

我还做了 benchmark，在不同 batch size 下比较 PyTorch 和 ONNX Runtime CPU 的平均推理延迟。这个 benchmark 很小，不代表真实工业性能，但能说明我理解了 latency、warmup、多次运行取平均和 batch size 对推理测试的影响。

和 CANN 的关系是：我现在完成的是 PyTorch -> ONNX -> ONNX Runtime CPU；后续在 Ascend 上可以理解为 ONNX -> ATC -> OM -> Ascend Runtime / CANN 执行。但我目前没有 Ascend 硬件和 CANN 环境，所以没有实际跑 ATC、OM 或 NPU 推理。
```

## 现场演示命令

```bash
cd /Users/lijingxuan/MyProject/onnx-inference-demo
.venv/bin/python src/infer_pytorch.py
.venv/bin/python src/export_onnx.py
.venv/bin/python src/infer_onnxruntime.py
.venv/bin/python src/benchmark.py
```

## 高频追问

### ONNX 是什么？

ONNX 是一种模型中间表示。它可以把 PyTorch 这类训练框架里的模型导出成通用格式，方便交给 ONNX Runtime、转换工具或硬件后端执行。它保存的不只是参数，也包括计算图、算子、输入输出信息。

### Runtime 是什么？

Runtime 是模型真正执行时依赖的运行系统。它负责加载模型、准备输入输出、调度计算图里的算子，并和具体执行后端交互。在这个项目里是 ONNX Runtime CPU。

### dummy_input 是什么？

dummy_input 是导出 ONNX 时使用的样例输入。PyTorch 会用它跑一遍 forward，从而追踪模型计算图、输入输出 shape 和算子关系。它不是训练数据，只是导出图结构时的示例。

### Gemm 和 Linear 的关系？

PyTorch 里的 Linear 本质上是矩阵乘法加 bias。导出到 ONNX 后，经常会表示成 Gemm 算子。我的模型是 Linear -> ReLU -> Linear，所以 ONNX 图里主要是 Gemm -> Relu -> Gemm。

### 没有 Ascend 环境，为什么还投 CANN？

我会把边界说清楚：我目前没有 Ascend 硬件和 CANN 环境，所以没有实际跑 ATC、OM 或 NPU 推理。但我已经从可验证的推理链路切入，理解 PyTorch、ONNX、Runtime、算子、计算图、latency，以及 ONNX -> ATC -> OM -> Ascend Runtime 的路径。我投这个方向，是希望从基础工程任务、文档、样例和工具链开始逐步深入。

### 不会 C++ 怎么解释？

我目前 C++ 还在补，不会把自己包装成已经能做复杂底层 C++ 开发的人。但我已经补了一个标准库版本的 C++ 算子 Demo，手写 MatMul、ReLU、Softmax，并用 std::chrono 看不同 batch size 下的耗时。它不是工业级高性能库，但能帮助我理解推理计算图里的基础算子在底层大概如何执行。

