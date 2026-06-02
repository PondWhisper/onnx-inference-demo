import onnxruntime as ort
import numpy as np


def main():
    session = ort.InferenceSession("model.onnx")

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print("input name", input_name)
    print("output name", output_name)

    x = np.random.randn(2, 4).astype(np.flaot32)

    print("input shape", x.shape)
    print("input dtype", x.dtype)

    outputs = session.run([output_name], {input_name: x})

    y = outputs[0]

    print("output", y)
    print("output shape", y.shape)
    print("output dtype", y.dtype)


if __name__ == "__main__":
    main()