"""
    Relu

Representation of an ONNX 'Relu' operator. 
Initialized from a protobuf descriptor object.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

from typing import List
import lib.onnx.onnx.onnx_ml_pb2 as onnx

import src.parser.meta.meta as meta

class Relu(meta.ONNXOperatorAttributes):
    def __init__(self, descriptor: List[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)
