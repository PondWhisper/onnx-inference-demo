# ONNX Inference Demo

本项目用于验证我是否适合继续探索 AI Infra / CANN / 模型推理方向。它从一个最小 PyTorch 模型开始，依次完成 ONNX 导出、ONNX Runtime 推理和简单 latency benchmark，并整理 CANN / Ascend 推理链路的学习笔记。

## 项目背景

我想通过一个小项目理解模型推理部署的基本流程，而不是只停留在概念层面。这个项目不追求复杂模型或高性能结果，而是追求把一条完整链路跑通：

```text
PyTorch 模型
-> 导出 ONNX
-> ONNX Runtime 推理
-> benchmark 测延迟
-> 理解 CANN / Ascend 在哪里
-> 整理成面试能讲的项目
```

## 技术路线

```text
PyTorch
  负责定义模型，并完成第一次本地推理

ONNX
  作为模型中间表示，让模型从 PyTorch 框架中导出

ONNX Runtime
  负责加载 ONNX 模型，并执行推理

Benchmark
  对 PyTorch 和 ONNX Runtime 的推理耗时做简单比较

CANN / Ascend
  当前项目不直接依赖 Ascend 硬件，但会理解 ONNX 到 OM、再到 Ascend 推理的部署链路
```

## 目录结构

```text
onnx-inference-demo/
├── README.md
├── requirements.txt
├── src/
│   ├── infer_pytorch.py
│   ├── export_onnx.py
│   ├── infer_onnxruntime.py
│   └── benchmark.py
├── notes/
│   ├── 01_pytorch_inference.md
│   ├── 02_onnx.md
│   ├── 03_runtime.md
│   ├── 04_benchmark.md
│   └── 05_cann_path.md
└── interview_qa.md
```

## 环境安装

建议使用虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 运行方式

第 1 步：运行 PyTorch 推理。

```bash
python src/infer_pytorch.py
```

第 2 步：导出 ONNX 模型。

```bash
python src/export_onnx.py
```

运行后会在项目根目录生成：

```text
model.onnx
```

第 3 步：使用 ONNX Runtime 推理。

```bash
python src/infer_onnxruntime.py
```

第 4 步：比较 PyTorch 和 ONNX Runtime 的平均推理耗时。

```bash
python src/benchmark.py
```

## 推理结果

`src/infer_pytorch.py` 会打印：

```text
input shape
output shape
output value
```

`src/infer_onnxruntime.py` 会打印：

```text
input name
output name
input shape
output shape
output value
```

输出数值不要求有业务含义，因为模型权重是随机初始化的。这个阶段重点是理解模型输入输出、shape 和推理执行过程。

## Benchmark 结果

`src/benchmark.py` 会分别运行 PyTorch 和 ONNX Runtime 推理多次，并打印平均 latency：

```text
PyTorch average latency:      x.xxxx ms
ONNX Runtime average latency: x.xxxx ms
```

这个 benchmark 只是入门实验，不代表真实生产性能。它的意义是让我开始关注推理场景里的 latency、warmup、repeat 次数和运行环境。

## CANN / Ascend 链路理解

当前项目跑通的是：

```text
PyTorch 模型
↓
ONNX 模型
↓
ONNX Runtime 推理
```

如果继续走向 Ascend / CANN 部署，链路通常会变成：

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

其中：

- Ascend 是华为昇腾 AI 处理器。
- CANN 是面向 Ascend 的异构计算架构和软件栈。
- ATC 是模型转换工具。
- OM 是 Ascend 侧可执行的离线模型格式。

## 项目局限

- 模型非常小，只用于理解流程。
- 没有训练过程，只关注推理。
- benchmark 没有覆盖真实 batch、真实数据和复杂模型。
- 当前没有 Ascend 硬件，因此没有实际完成 ONNX -> OM 和 Ascend 推理。
- 对 CANN 的理解目前停留在部署链路和概念层面，还需要继续补充算子、内存、图优化和硬件执行相关知识。

## 下一步

- 使用一个真实模型，例如 MNIST MLP、ResNet 或 MobileNet。
- 对比不同 batch size 下的 latency 和 throughput。
- 学习 ONNX 模型结构，查看计算图和算子。
- 继续学习 CANN、ATC、OM、Ascend Runtime。
- 如果有 Ascend 环境，补充 ONNX -> OM -> Ascend 推理实验。
