# CANN / Ascend Path

## 1. Ascend 是什么？

Ascend 是华为的 AI 处理器 / NPU 平台，用来执行 AI 训练或推理任务。

## 2. CANN 是什么？

CANN 是面向 Ascend AI 处理器的软件栈 / 异构计算架构。它负责连接上层 AI 框架、模型表示和底层昇腾硬件。

## 3. CANN Execution Provider 是什么？

CANN Execution Provider 是 ONNX Runtime 里的一个执行后端。普通 ONNX Runtime 可以用 CPUExecutionProvider 在 CPU 上执行模型；如果有 Ascend 和 CANN 环境，就可以通过 CANNExecutionProvider 尝试在 Ascend 硬件上执行 ONNX 模型。

## 4. 它和我前面的项目有什么关系？

我前面已经完成了 PyTorch → ONNX → ONNX Runtime CPU 推理。CANN EP 可以理解为把 ONNX Runtime 的执行后端从 CPU 换成 Ascend / CANN。

## 5. ATC 是什么？

ATC 是 CANN 提供的模型转换工具。它可以把 ONNX 模型转换成 Ascend AI 处理器可以识别的 OM 模型。

## 6. OM 模型是什么？

OM 模型可以理解为 Ascend 侧用于推理执行的离线模型格式。ONNX 是通用模型格式，OM 是面向昇腾推理环境的模型格式。

## 7. 我的项目如果接到 Ascend，大概链路是什么？

PyTorch 模型
↓ torch.onnx.export
ONNX 模型 model.onnx
↓ ATC
OM 模型 simple_model.om
↓ AscendCL / Runtime
Ascend NPU 推理

 ## 8. 当前学习边界

我目前没有 Ascend 硬件和 CANN 环境，所以没有实际运行 ATC 转换，也没有在 Ascend NPU 上做推理测试。当前项目完成的是 CPU 环境下的 PyTorch → ONNX → ONNX Runtime 推理闭环，并通过官方文档理解 ONNX → OM → Ascend 推理链路。