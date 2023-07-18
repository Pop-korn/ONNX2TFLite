"""
    Dropout

Representation of an ONNX 'Dropout' operator. 
Initialized from a protobuf descriptor object.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

from typing import List
import lib.onnx.onnx.onnx_ml_pb2 as onnx

import src.parser.meta.meta as meta

import src.err as err

class Dropout(meta.ONNXOperatorAttributes):
    # Attribute is 'None' if not present in the model
    seed: int

    def __init__(self, descriptor: List[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _defaultValues(self):
        self.seed = None

    def _initAttributes(self):
        for attr in self._descriptor:
            if attr.name == "seed": # Not tested!
                self.seed = attr.i
            else:
                err.warning(f"ONNX Dropout attribute '{attr.name}' is not supported!")
