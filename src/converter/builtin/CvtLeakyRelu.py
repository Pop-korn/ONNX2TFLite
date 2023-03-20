import src.parser.builtin.LeakyRelu as onnxLR

import src.generator.builtin.LeakyRelu as tflLR

def convert(oLR: onnxLR.LeakyRelu) -> tflLR.LeakyRelu:
    return tflLR.LeakyRelu(oLR.alpha)
