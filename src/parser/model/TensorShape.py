"""
    TensorShape

Representation of ONNX 'TensorShape' objects.
Initialized from a protobuf descriptor.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

from typing import List, Union
import lib.onnx.onnx.onnx_ml_pb2 as onnx

import src.parser.meta.meta as meta

import src.err as err

class Dimention(meta.ONNXObject):
    value: Union[int, str]
    denotation: str

    def __init__(self, descriptor: onnx.TensorShapeProto.Dimension) -> None:
        super().__init__(descriptor)
        self.denotation = descriptor.denotation

        if descriptor.HasField("dim_value"):
            self.value = descriptor.dim_value
        elif descriptor.HasField("dim_param"):
            self.value = descriptor.dim_param
        else:
            err.warning("ONNX TensorShape.Dimension has no valid value!")

class TensorShape(meta.ONNXObject):
    dims: List[Dimention]

    def __init__(self, descriptor: onnx.TensorShapeProto) -> None:
        super().__init__(descriptor)
        self.dims = []
        for item in descriptor.dim:
            self.dims.append(Dimention(item))