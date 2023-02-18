import src.parser.model.Tensor as onnxTensor

import src.generator.model.Tensors as tflTnsors


def convertTensors(tflTensor: onnxTensor.Tensor):
    print(tflTensor.name)