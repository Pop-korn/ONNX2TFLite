"""
    Conv

Representation of an ONNX 'Conv' operator. 
Initialized from a protobuf descriptor object.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

from typing import List
import lib.onnx.onnx.onnx_ml_pb2 as onnx

import src.parser.meta.meta as meta

import src.err as err

class Conv(meta.ONNXOperatorAttributes):
    # Attribute is 'None' if not present in the model
    autoPad: str
    dilations: meta.ONNXIntListAttribute
    group: int
    kernelShape :meta.ONNXIntListAttribute
    pads: meta.ONNXIntListAttribute
    strides: meta.ONNXIntListAttribute

    def __init__(self, descriptor: List[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _defaultValues(self):
        self.autoPad = "NOTSET"
        self.dilations = None
        self.group = 1
        self.kernelShape = None
        self.pads = None
        self.strides = None

    def _initAttributes(self):
        for attr in self._descriptor:
            if attr.name == "auto_pad":
                self.autoPad = attr.s # Not tested!
            elif attr.name == "dilations":
                self.dilations = meta.ONNXIntListAttribute(attr) # Not tested!
            elif attr.name == "group":
                self.group = attr.i
            elif attr.name == "kernel_shape":
                self.kernelShape = meta.ONNXIntListAttribute(attr)
            elif attr.name == "pads":
                self.pads = meta.ONNXIntListAttribute(attr)
            elif attr.name == "strides":
                self.strides = meta.ONNXIntListAttribute(attr)
            else:
                err.warning(f"ONNX Conv attribute '{attr.name}' is not supported!")
