# ONNX

## 1. ONNX 是什么？

ONNX 是一种模型中间格式。它可以把 PyTorch 里的模型导出成通用模型文件，让其他 Runtime 或硬件后端读取和执行。

## 2. 为什么要把 PyTorch 模型导出为 ONNX？

因为模型不一定只在 PyTorch 里运行。导出 ONNX 后，可以交给 ONNX Runtime 或其他后端执行，也方便后续适配不同硬件。

## 3. dummy_input 有什么用？

dummy_input 是一个样例输入。PyTorch 会用它跑一遍 forward，从而追踪模型的计算图、输入 shape 和输出 shape。

## 4. input_names / output_names 有什么用？

它们给 ONNX 模型的输入和输出命名。后面使用 ONNX Runtime 时，需要通过这些名字把输入喂给模型，并获取输出。

## 5. opset_version 是什么？

opset_version 表示 ONNX 算子版本。这里先使用 17，不需要深入研究。

## 6. model.onnx 里面大概保存了什么？

model.onnx 里保存了模型的计算图、参数和输入输出信息。它不是 Python 源码，而是可以被 ONNX Runtime 等工具读取的模型文件。

## 7. 用 Netron 看 model.onnx 后，我看到了什么？

我看到 ONNX 模型里有输入节点、输出节点和中间计算节点。

输入节点名称是 input，对应导出 ONNX 时设置的 input_names=["input"]。

输出节点名称是 output，对应 output_names=["output"]。

模型中间主要包含 Gemm / Relu / Gemm 这类节点。PyTorch 中的 Linear 层在 ONNX 中通常会被表示为 Gemm 或 MatMul + Add，ReLU 会被表示为 Relu 算子。

我还看到了一些 weight 和 bias，这说明 ONNX 文件不只是保存模型结构，也保存了模型参数。

## 8. 什么是算子？

算子是模型里的基本计算单元，比如 Linear、MatMul、Gemm、Relu、Conv、Softmax 等。模型推理时，输入数据会依次经过这些算子，最终得到输出。

## 9. 什么是计算图？

计算图描述了数据从输入到输出的计算路径。比如我的模型可以理解为：

input [1, 4]
↓
Gemm / Linear
↓
Relu
↓
Gemm / Linear
↓
output [1, 2]

ONNX 文件保存的就是这种计算图和模型参数。