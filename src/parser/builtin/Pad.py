"""
    Pad

Representation of an ONNX 'Pad' operator. 
Initialized from a protobuf descriptor object.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

from typing import List
import lib.onnx.onnx.onnx_ml_pb2 as onnx

import src.parser.meta.meta as meta

import src.err as err

class Pad(meta.ONNXOperatorAttributes):
    mode: str
    pads: List[int]
    value: float

    def __init__(self, descriptor: List[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _defaultValues(self):
        self.mode = "constant"
        self.pads = []
        self.value = 0.0

    def _initAttributes(self):
        for attr in self._descriptor:
            match attr.name:
                case "mode":
                    self.mode = attr.s
                case "pads":
                    self.pads = attr.ints
                case "value":
                    self.value = attr.f
                case _:
                    err.warning(f"ONNX Pad attribute '{attr.name}' is not supported!")

