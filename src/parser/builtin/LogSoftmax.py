from typing import List

import lib.onnx.onnx.onnx_ml_pb2 as onnx

import src.parser.meta.meta as meta

import src.err as err

class LogSoftmax(meta.ONNXOperatorAttributes):
    # Attribute is 'None' if not present in the model
    axis: int

    def __init__(self, descriptor: List[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _defaultValues(self):
        self.axis = -1
            
    def _initAttributes(self):
        for attr in self._descriptor:
            match attr.name:
                case "axis":
                    self.axis = attr.i 
                case _:
                    err.warning(f"ONNX Softmax attribute '{attr.name}' is not supported!")
                    