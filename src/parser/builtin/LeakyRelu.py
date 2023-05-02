"""
    LeakyRelu

Representation of an ONNX 'LeakyRelu' operator. 
Initialized from a protobuf descriptor object.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

from typing import List

import src.parser.meta.meta as meta

import src.err as err

import lib.onnx.onnx.onnx_ml_pb2 as onnx

class LeakyRelu(meta.ONNXOperatorAttributes):
    alpha: float

    def __init__(self, descriptor: List[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _defaultValues(self):
        self.alpha = 0.01

    def _initAttributes(self):
        for attr in self._descriptor:
            if attr.name == "alpha":
                self.alpha = attr.f
            else:
                err.warning(f"ONNX LeakyRelu attribute '{attr.name}' is not supported!")
