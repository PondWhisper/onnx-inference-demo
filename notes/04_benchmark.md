# Benchmark

## 1. latency 是什么？

latency 是单次推理延迟，也就是模型处理一次输入平均需要多久。

## 2. throughput 是什么？

throughput 是吞吐量，也就是单位时间能处理多少请求或样本。

## 3. warmup 是什么？

warmup 是正式计时前先运行几次模型，避免第一次运行的初始化开销影响 benchmark 结果。

## 4. 为什么 benchmark 要跑多次？

因为单次运行容易受系统状态、初始化、CPU 调度等因素影响。跑多次取平均值会更稳定。

## 5. 这个实验有什么局限？

这个模型太小，benchmark 结果可能波动很大，不能代表真实大模型或工业场景性能。当前只在 CPU 上比较 PyTorch 和 ONNX Runtime，没有测试 GPU、NPU 或 CANN 后端。