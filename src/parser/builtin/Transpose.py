from typing import List
import lib.onnx.onnx.onnx_ml_pb2 as onnx

import src.parser.meta.meta as meta

import src.err as err

class Transpose(meta.ONNXOperatorAttributes):
    perm: List[int]

    def __init__(self, descriptor: List[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _initAttributes(self):
        for attr in self._descriptor:
            match attr.name:
                case "perm":
                    self.perm = attr.ints
                case _:
                    err.warning(f"ONNX Transpose attribute '{attr.name}' is not supported!")
