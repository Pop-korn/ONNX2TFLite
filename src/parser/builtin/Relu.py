import onnx.onnx.onnx_ml_pb2 as onnx

import parser.meta.meta as meta

class Relu(meta.ONNXOperatorAttributes):
    def __init__(self, descriptor: list[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)