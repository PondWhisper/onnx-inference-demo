# 02 ONNX

这一步对应 `src/export_onnx.py`，目标是把 PyTorch 模型导出成 `model.onnx`。

ONNX 是 Open Neural Network Exchange 的缩写，可以理解成一种模型中间格式。PyTorch、TensorFlow 等训练框架各有自己的模型表示，如果模型只能在原框架里运行，部署会受到限制。导出成 ONNX 后，模型就可以交给 ONNX Runtime 或其他推理引擎执行。

`torch.onnx.export` 的作用是把 PyTorch 模型和一次示例输入一起转换成 ONNX 计算图。这里的示例输入叫 `dummy_input`，它不一定是真实业务数据，但它告诉导出工具输入的 shape 和 dtype。导出工具会根据模型结构和 dummy input 跟踪计算过程，生成一张静态或半静态的计算图。

`input_names` 和 `output_names` 用来给模型输入输出命名，方便 Runtime 执行时绑定数据。`opset_version` 表示 ONNX 算子集合版本。算子可以理解成计算图里的基础计算单元，例如 `Linear` 最终会被拆成矩阵乘法、加法等算子。计算图则描述这些算子之间的数据流关系。

这一步的意义是：模型不再只属于 PyTorch，而是被转换成一个更通用的中间表示。对 CANN / Ascend 方向来说，ONNX 很重要，因为很多部署链路会先把训练框架模型导出成 ONNX，再继续转换成硬件侧可执行的格式。
