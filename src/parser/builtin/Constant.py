"""
    Constant

Representation of an ONNX 'Constant' operator. 
Initialized from a protobuf descriptor object.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

from typing import List

import lib.onnx.onnx.onnx_ml_pb2 as onnx

import src.parser.meta.meta as meta

import src.parser.model.Tensors as Tensors

import src.err as err

class Constant(meta.ONNXOperatorAttributes):
    value: Tensors.Tensor
    # TODO add other value options

    def __init__(self, descriptor: List[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _initAttributes(self):
        for attr in self._descriptor:
            if attr.name == "value":
                self.value = Tensors.Tensor(attr.t)
            else:
                err.error(err.Code.UNSUPPORTED_OPERATOR_ATTRIBUTES,
                          f"ONNX Operator 'Constant' has attribute '{attr.name}'",
                          "which is not yet supported!")
