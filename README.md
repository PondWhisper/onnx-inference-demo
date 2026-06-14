# ONNX Inference Demo

这个项目用于整理一个最小模型推理部署链路：从 PyTorch 定义并运行模型，到导出 ONNX，再到使用 ONNX Runtime 在 CPU 上执行推理，并进一步理解它和 CANN / Ascend 推理链路之间的关系。

项目目标不是做复杂模型，而是证明我理解模型从训练框架走向推理 Runtime 的基本过程，并能把这个过程整理成面试官能看懂的项目说明。

## 项目背景

在模型训练阶段，PyTorch 很适合做模型定义、调试和训练。但真实部署时，模型往往需要脱离训练框架，交给专门的推理 Runtime 或硬件后端执行。因此我用这个项目跑通一条基础链路：

```text
PyTorch model
-> ONNX model
-> ONNX Runtime inference
-> Benchmark
-> CANN / Ascend inference path understanding
```

这条链路可以帮助我理解：

- PyTorch 模型如何完成最小推理。
- 为什么要把模型导出为 ONNX。
- Runtime 如何加载模型并执行计算图。
- 推理场景为什么关注 latency 和 throughput。
- CANN / Ascend 部署链路中 ONNX、ATC、OM 的位置。

## 技术路线

```text
PyTorch
  定义 SimpleModel，并完成一次最小推理。

ONNX
  将 PyTorch 模型导出为通用中间表示 model.onnx。

ONNX Runtime
  在 CPU 上加载 model.onnx，并用 NumPy 输入执行推理。

Benchmark
  比较 PyTorch 和 ONNX Runtime 的平均推理延迟。

CANN / Ascend
  基于当前 ONNX Demo，理解 PyTorch -> ONNX -> ATC -> OM -> Ascend/CANN 的部署路径。
```

## 目录结构

```text
onnx-inference-demo/
├── README.md
├── requirements.txt
├── src/
│   ├── infer_pytorch.py
│   ├── save_load_pytorch.py
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

## Progress

- [x] Run PyTorch minimal inference
- [x] Save and load PyTorch model
- [x] Export PyTorch model to ONNX
- [x] Visualize ONNX graph
- [x] Run inference with ONNX Runtime
- [x] Benchmark PyTorch vs ONNX Runtime across different batch sizes
- [x] Understand CANN / Ascend inference path

## 环境安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果只运行 ONNX Runtime 推理，至少需要：

```bash
pip install onnxruntime numpy
```

## 运行方式

运行 PyTorch 最小推理：

```bash
python src/infer_pytorch.py
```

保存并加载 PyTorch 模型：

```bash
python src/save_load_pytorch.py
```

导出 ONNX 模型：

```bash
python src/export_onnx.py
```

运行后会在项目根目录生成：

```text
model.onnx
```

使用 ONNX Runtime 推理：

```bash
python src/infer_onnxruntime.py
```

运行 benchmark：

```bash
python src/benchmark.py
```

本机项目已包含 `.venv`，面试现场演示可以直接运行：

```bash
.venv/bin/python src/infer_pytorch.py
.venv/bin/python src/export_onnx.py
.venv/bin/python src/infer_onnxruntime.py
.venv/bin/python src/benchmark.py
```

## ONNX Runtime Inference

```bash
python src/infer_onnxruntime.py
```

This script loads `model.onnx` with ONNX Runtime and runs inference with a NumPy input.

ONNX Runtime 学习块完成标准：

```text
1. pip install onnxruntime numpy 成功
2. src/infer_onnxruntime.py 能运行
3. ONNX Runtime 输出 shape 是 [1, 2]
4. 完成 shape 错误实验
5. 完成 dtype 错误实验
6. notes/03_runtime.md 写完
```

## 推理结果

PyTorch 推理会打印模型输入输出信息，例如：

```text
input shape
output shape
output value
```

ONNX Runtime 推理会打印模型输入名、输出名、shape 和输出值，例如：

```text
input name
output name
input shape
output shape
output value
```

当前模型是一个很小的全连接网络，权重没有训练过，所以输出值本身没有业务含义。这个阶段重点是确认模型能被正确加载、输入 shape 正确、推理链路能跑通。

## Benchmark 结果

`src/benchmark.py` 会在不同 batch size 下分别测试 PyTorch 和 ONNX Runtime CPU 的平均推理延迟。每个 batch size 先 warmup 20 次，再计时运行 1000 次。

最近一次本机运行结果：

| Batch size | PyTorch avg latency (ms) | ONNX Runtime CPU avg latency (ms) |
| ---: | ---: | ---: |
| 1 | 0.007437 | 0.002461 |
| 8 | 0.007704 | 0.003107 |
| 32 | 0.007559 | 0.002929 |
| 128 | 0.009327 | 0.003753 |

这个 benchmark 只是入门实验，不代表真实生产性能。模型很小，耗时可能主要来自 Python 调用、框架调度和 Runtime 开销。它的意义是让我开始关注 latency、warmup、重复运行、batch size 和运行环境这些推理性能测试里的基本问题。

## ONNX 图里的算子

当前模型结构是：

```text
Linear(4 -> 8)
ReLU
Linear(8 -> 2)
```

导出到 ONNX 后，图里的主要算子是：

```text
Gemm
Relu
Gemm
```

其中：

- `Gemm` 可以理解为全连接层里的矩阵乘法加 bias，对应 PyTorch 里的 `Linear`。
- `Relu` 对应 PyTorch 里的 `ReLU` 激活函数。
- ONNX 文件不仅保存模型结构，也保存 weight、bias 等参数。

这说明 PyTorch 里的模型层在导出后会变成 ONNX 计算图中的算子节点。Runtime 执行模型时，本质上是在按照计算图依次执行这些算子。

## CANN / Ascend 推理链路

This project currently runs ONNX inference with ONNX Runtime on CPU.

Based on CANN / Ascend documentation, a possible Ascend deployment path is:

```text
PyTorch model
-> ONNX model
-> ATC conversion
-> OM model
-> Ascend / CANN inference
```

对应关系：

- `Ascend` 是华为昇腾 AI 处理器。
- `CANN` 是面向 Ascend 的 AI 计算软件栈。
- `ATC` 是模型转换工具，可以把 ONNX 等模型转换为 Ascend 可执行的 OM 模型。
- `OM` 是 Ascend 侧用于推理执行的离线模型格式。

当前项目已经实现的是：

```text
PyTorch
-> ONNX
-> ONNX Runtime CPU inference
```

当前项目没有实际实现的是：

```text
ONNX
-> ATC
-> OM
-> Ascend / CANN inference
```

## 当前局限

- 没有 Ascend 硬件。
- 没有 CANN 开发和运行环境。
- 没有实际运行 ATC 模型转换。
- 没有生成 OM 模型。
- 没有在 Ascend NPU 上执行推理。
- 当前 benchmark 只计划比较 CPU 上的 PyTorch 和 ONNX Runtime，不能代表 Ascend/CANN 性能。

因此，这个项目的边界是：我完成了 PyTorch -> ONNX -> ONNX Runtime CPU 推理链路，并把 Ascend/CANN 推理路径作为学习笔记整理出来，但还没有完成真实 Ascend 部署。
