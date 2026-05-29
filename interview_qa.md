# Interview QA

## 1. 你为什么投 CANN？

我想从模型应用继续往更底层的 AI Infra 和推理部署方向走。CANN 位于模型和昇腾硬件之间，涉及模型转换、算子、Runtime、性能优化和硬件适配，这些内容比单纯调 API 更接近 AI 系统的核心。我做这个 ONNX Inference Demo，就是为了先把 PyTorch -> ONNX -> Runtime -> benchmark 这条基础链路跑通，再继续理解 ONNX -> ATC -> OM -> Ascend 推理。

## 2. 训练和推理有什么区别？

训练是让模型通过数据和 loss 更新参数，需要前向传播、反向传播和优化器更新。推理是不更新参数，只把输入送进已经确定的模型，得到输出结果。训练更关注收敛和精度，推理更关注 latency、throughput、资源占用和部署稳定性。

## 3. Tensor / shape / batch size 是什么？

Tensor 是深度学习框架里的多维数组。shape 描述 Tensor 每个维度的大小，例如 `(1, 4)` 表示 1 条样本，每条样本有 4 个特征。batch size 是一次送进模型的样本数量，通常是 shape 的第一个维度。

## 4. ONNX 是什么？

ONNX 是一种模型中间表示。它可以把 PyTorch 等训练框架里的模型导出成更通用的格式，方便交给不同推理引擎或硬件后端执行。

## 5. 为什么要导出 ONNX？

因为生产部署时不一定希望模型只依赖原训练框架。导出 ONNX 后，模型可以交给 ONNX Runtime、TensorRT、CANN 等推理或转换工具处理，更方便跨框架、跨硬件部署。

## 6. Runtime 是什么？

Runtime 是模型执行时依赖的运行系统。它负责加载模型、准备输入输出、调度计算图、执行算子，并和具体硬件后端交互。ONNX Runtime 就是执行 ONNX 模型的 Runtime。

## 7. 算子和计算图是什么？

算子是模型里的基础计算单元，例如矩阵乘法、加法、ReLU。计算图描述这些算子之间的数据流和依赖关系。模型推理时，Runtime 会按照计算图依次或并行执行算子。

## 8. latency / throughput 是什么？

Latency 是一次推理请求的耗时，通常用毫秒表示。Throughput 是单位时间内能处理多少请求或样本。在线服务通常非常关注 latency，离线批处理则会更关注 throughput。

## 9. CANN / Ascend / ATC / OM 是什么关系？

Ascend 是华为昇腾 AI 处理器。CANN 是面向 Ascend 的 AI 计算软件栈。ATC 是 CANN 里的模型转换工具，可以把 ONNX 等模型转成 Ascend 可执行的 OM 模型。OM 模型再由 Ascend Runtime 加载，在 Ascend 硬件上执行推理。

## 10. 你不会 C++，为什么还投这个岗位？

我现在的 C++ 经验确实不足，所以不会夸大自己。但我已经开始从模型推理链路切入，先用 Python 跑通 PyTorch、ONNX、ONNX Runtime 和 benchmark，理解 Runtime、算子、计算图、模型转换这些基础概念。我的计划是先把部署链路理解清楚，再补 C++ 和系统层能力。我投这个方向，是因为我愿意从底层和工程问题长期学习。

## 11. 这个项目有什么局限？

第一，模型很小，只能证明链路跑通，不能代表真实业务模型。第二，benchmark 很简单，没有覆盖多 batch、多线程、真实数据和复杂硬件环境。第三，目前没有 Ascend 硬件，所以还没有实际完成 ONNX -> OM 和 CANN 推理。这个项目更像是进入 AI Infra / CANN 方向的第一步。

## 12. 你怎么介绍这个项目？

我做了一个 ONNX 推理链路 Demo。它先用 PyTorch 定义一个简单模型并执行推理，然后用 `torch.onnx.export` 导出成 ONNX，再用 ONNX Runtime 加载 ONNX 模型执行推理，最后比较 PyTorch 和 ONNX Runtime 的平均 latency。通过这个项目，我理解了 Tensor、shape、计算图、算子、Runtime 和 benchmark 的基础概念，也把它和 CANN 场景里的 PyTorch -> ONNX -> ATC -> OM -> Ascend 推理链路联系起来。
