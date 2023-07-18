"""
    Gemm

Representation of an ONNX 'Gemm' operator. 
Initialized from a protobuf descriptor object.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

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
            if attr.name ==  "alpha":
                self.alpha = attr.f # Not tested!
            elif attr.name ==  "beta":
                self.beta = attr.f # Not tested!
            elif attr.name ==  "transA":
                self.transA = attr.i # Not tested!
            elif attr.name ==  "transB":
                self.transB = attr.i
            else:
                err.warning(f"ONNX Gemm attribute '{attr.name}' is not supported!")
