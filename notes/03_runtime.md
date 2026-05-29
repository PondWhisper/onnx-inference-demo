# 03 Runtime

这一步对应 `src/infer_onnxruntime.py`，目标是用 ONNX Runtime 加载 `model.onnx` 并执行推理。

Runtime 可以理解成模型真正执行时所依赖的运行系统。训练框架主要服务于模型开发和训练，而推理 Runtime 更关注模型加载、图执行、算子调度、内存管理和硬件后端适配。ONNX Runtime 就是一个可以执行 ONNX 模型的推理 Runtime。

`InferenceSession` 是 ONNX Runtime 的核心对象。创建 session 时，Runtime 会读取 ONNX 模型，检查模型结构，并准备执行计划。之后通过 `session.get_inputs()` 和 `session.get_outputs()` 可以拿到模型输入输出信息，包括名称、shape 和 dtype。真正推理时，调用 `session.run([output_name], {input_name: input_array})`，Runtime 会把 numpy 输入送入计算图，然后返回输出。

ONNX Runtime 能执行 ONNX 模型，是因为 ONNX 文件里保存了计算图和算子信息。Runtime 根据这些信息找到对应算子的实现，再按照图里的依赖关系执行。不同 provider 可以把计算放到不同硬件上，例如 CPU、CUDA GPU 或其他后端。

这一步让我理解到：模型部署并不是“保存模型文件”这么简单。真正上线时，需要一个 Runtime 去解释或编译模型、分配内存、调度算子，并和硬件后端打交道。CANN 在 Ascend 场景里也承担类似的底层执行和硬件适配角色。
