# Interview QA

## 1. 你为什么投 CANN？

我想从模型应用继续往 AI Infra 和推理部署方向走。CANN 位于模型和 Ascend 硬件之间，会涉及模型转换、Runtime、算子、图执行和性能优化，这些内容比单纯调用模型 API 更接近 AI 系统底层。

我做这个 ONNX Inference Demo，是为了先跑通 PyTorch -> ONNX -> ONNX Runtime CPU 这条基础推理链路，再继续理解 ONNX -> ATC -> OM -> Ascend/CANN 推理。这个项目不是为了证明我已经精通 CANN，而是证明我在沿着这个方向拆解和学习。

## 2. 训练和推理有什么区别？

训练是让模型通过数据和 loss 更新参数，通常包括前向传播、计算 loss、反向传播和优化器更新。推理是不更新参数，只把输入送进已经确定的模型，得到输出结果。

训练更关注模型能不能收敛、精度能不能提升；推理更关注 latency、throughput、资源占用、稳定性和部署成本。

## 3. ONNX 是什么？

ONNX 是一种模型中间表示，可以把 PyTorch 等训练框架里的模型导出成更通用的格式。这样模型就不一定只能在原来的训练框架里运行，也可以交给 ONNX Runtime 或其他推理、转换工具处理。

在我的项目里，ONNX 的作用是把 PyTorch 里的 `SimpleModel` 导出成 `model.onnx`，让后续的 ONNX Runtime 可以加载和执行。

## 4. Runtime 是什么？

Runtime 是模型真正执行时依赖的运行系统。它负责加载模型、准备输入输出、执行计算图、调度算子，并和具体硬件或执行后端交互。

在我的项目里，ONNX Runtime 会加载 `model.onnx`，读取模型 input/output 信息，然后用 NumPy 输入执行推理。它说明模型已经不再依赖 PyTorch forward，而是交给一个推理 Runtime 来执行。

## 5. 算子和计算图是什么？

算子是模型里的基础计算单元，比如 `Gemm`、`Relu`、`Conv`、`Softmax`。计算图描述这些算子之间的数据流关系，也就是输入先经过哪个算子，再进入哪个算子，最后得到输出。

我的 ONNX 图里主要是 `Gemm / Relu / Gemm`。这对应 PyTorch 模型里的 `Linear / ReLU / Linear`。Runtime 执行模型时，本质上就是按照计算图执行这些算子。

## 6. latency / throughput 是什么？

Latency 是一次推理从开始到结束的耗时，通常用毫秒表示。Throughput 是单位时间内能处理多少请求或样本，通常可以理解为每秒处理量。

在线推理服务通常很关注 latency，因为用户希望单次请求尽快返回；批处理或高并发场景也会关注 throughput，因为它代表系统整体处理能力。

## 7. CANN / Ascend / ATC / OM 是什么关系？

Ascend 是华为昇腾 AI 处理器。CANN 是面向 Ascend 的 AI 计算软件栈，负责连接上层模型和底层硬件执行。ATC 是 CANN 里的模型转换工具，可以把 ONNX 等模型转换成 Ascend 可执行的 OM 模型。OM 是 Ascend 侧用于推理执行的离线模型格式。

我理解的部署链路是：PyTorch model -> ONNX model -> ATC conversion -> OM model -> Ascend/CANN inference。

## 8. 没有 Ascend 环境，你这个项目的边界是什么？

这个项目的边界很明确：我目前没有 Ascend 硬件，也没有 CANN 环境，所以没有实际运行 ATC 转换，没有生成 OM 模型，也没有在 Ascend NPU 上做推理。

我真正完成的是 PyTorch -> ONNX -> ONNX Runtime CPU 推理链路，并通过笔记整理了 ONNX -> ATC -> OM -> Ascend/CANN 的后续路径。面试中我会把它定位成 CANN 方向的入门推理链路 Demo，而不是完整 Ascend 部署项目。

## 9. 你不会 C++，为什么还投这个方向？

我目前 C++ 经验确实不足，所以不会把自己包装成已经能做底层 C++ 开发的人。但我已经开始从推理链路切入，理解 PyTorch、ONNX、Runtime、算子、计算图、latency 和 CANN/Ascend 部署路径这些基础概念。

我投这个方向，是因为我愿意补系统能力和 C++ 能力，也愿意从小实验开始逐步靠近底层工程问题。这个项目证明的是我的学习路径和方向感：先跑通可验证的推理链路，再继续补 CANN、Ascend Runtime、算子和 C++。
