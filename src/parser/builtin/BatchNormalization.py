
from typing import List

import lib.onnx.onnx.onnx_ml_pb2 as onnx

import src.parser.meta.meta as meta

class BatchNormalization(meta.ONNXOperatorAttributes):
    epsilon: float
    # Other attributes are only used while training. So there is no need to
    # convert them

    def __init__(self, descriptor: List[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _defaultValues(self):
        self.epsilon = 1e-5

    def _initAttributes(self):
        for attr in self._descriptor:
            if attr.name == "epsilon":
                self.epsilon = attr.f
