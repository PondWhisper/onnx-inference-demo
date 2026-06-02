# CANN / Ascend Path

## 1. Ascend 是什么？

Ascend 是华为的 AI 处理器 / NPU 平台，用来执行 AI 训练或推理任务。

## 2. CANN 是什么？

CANN 是面向 Ascend AI 处理器的软件栈 / 异构计算架构。它负责连接上层 AI 框架、模型表示和底层昇腾硬件。

## 3. CANN Execution Provider 是什么？

CANN Execution Provider 是 ONNX Runtime 里的一个执行后端。普通 ONNX Runtime 可以用 CPUExecutionProvider 在 CPU 上执行模型；如果有 Ascend 和 CANN 环境，就可以通过 CANNExecutionProvider 尝试在 Ascend 硬件上执行 ONNX 模型。

## 4. 它和我前面的项目有什么关系？

我前面已经完成了 PyTorch → ONNX → ONNX Runtime CPU 推理。CANN EP 可以理解为把 ONNX Runtime 的执行后端从 CPU 换成 Ascend / CANN。