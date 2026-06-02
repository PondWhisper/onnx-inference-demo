# ONNX Runtime

## 1. Runtime 是什么？

Runtime 可以理解为执行模型的运行环境。PyTorch 可以执行 PyTorch 模型，ONNX Runtime 可以执行 ONNX 模型。

## 2. ONNX Runtime 是什么？

ONNX Runtime 是用来执行 ONNX 模型的推理引擎。它可以加载 model.onnx，接收输入，然后执行计算图，得到输出。

## 3. InferenceSession 是什么？

InferenceSession 是 ONNX Runtime 创建的模型执行会话。它负责加载 ONNX 模型，并准备执行推理。

## 4. session.get_inputs() 和 session.get_outputs() 有什么用？

它们用来读取 ONNX 模型的输入和输出信息，比如输入名、输出名、shape、dtype 等。

## 5. session.run() 做了什么？

session.run() 会把输入数据喂给 ONNX 模型，然后执行计算图，最后返回模型输出。

## 6. 为什么 ONNX Runtime 输入要用 numpy.float32？

因为 ONNX Runtime 的 Python API 通常接收 NumPy array。模型导出时使用的是 float32，所以输入也要是 float32，否则可能会出现类型不匹配。