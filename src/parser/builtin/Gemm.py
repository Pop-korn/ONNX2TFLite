from typing import List
import lib.onnx.onnx.onnx_ml_pb2 as onnx

import src.parser.meta.meta as meta

import src.err as err

class Gemm(meta.ONNXOperatorAttributes):
    # Attribute is 'None' if not present in the model
    alpha: float
    beta: float
    transA: int
    transB: int

    def __init__(self, descriptor: List[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _defaultValues(self):
        self.alpha = 1.0
        self.beta = 1.0
        self.transA = 0
        self.transB = 0

    def _initAttributes(self):
        for attr in self._descriptor:
            match attr.name:
                case "alpha":
                    self.alpha = attr.f # Not tested!
                case "beta":
                    self.beta = attr.f # Not tested!
                case "transA":
                    self.transA = attr.i # Not tested!
                case "transB":
                    self.transB = attr.i
                case _:
                    err.warning(f"ONNX Gemm attribute '{attr.name}' is not supported!")
