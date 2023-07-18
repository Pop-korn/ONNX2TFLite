"""
    LRN

Representation of an ONNX 'LRN' operator. 
Initialized from a protobuf descriptor object.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

from typing import List
import lib.onnx.onnx.onnx_ml_pb2 as onnx

import src.parser.meta.meta as meta

import src.err as err

class LRN(meta.ONNXOperatorAttributes):
    alpha: float
    beta: float
    bias: float
    size: int

    def __init__(self, descriptor: List[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _defaultValues(self):
        self.alpha = 0.0001
        self.beta = 0.75
        self.bias = 1.0
        self.size = None

    def _initAttributes(self):
        for attr in self._descriptor:
            if attr.name ==  "alpha":
                self.alpha = attr.f
            elif attr.name ==  "beta":
                self.beta = attr.f
            elif attr.name ==  "bias":
                self.bias = attr.f
            elif attr.name ==  "size":
                self.size = attr.i
            else:
                err.warning(f"ONNX LRN attribute '{attr.name}' is not supported!")
