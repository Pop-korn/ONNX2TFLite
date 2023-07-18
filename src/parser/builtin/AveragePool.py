"""
    AveragePool

Representation of an ONNX 'AveragePool' operator. 
Initialized from a protobuf descriptor object.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

from typing import List
import lib.onnx.onnx.onnx_ml_pb2 as onnx

import src.parser.meta.meta as meta

import src.err as err

class AveragePool(meta.ONNXOperatorAttributes):
    # Attribute is 'None' if not present in the model
    autoPad: str
    ceilMode: int
    countIncludePad: int
    dilations: meta.ONNXIntListAttribute
    kernelShape :meta.ONNXIntListAttribute
    pads: meta.ONNXIntListAttribute
    strides: meta.ONNXIntListAttribute

    def __init__(self, descriptor: List[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)
        self.__defaultValues()
        self.__initAttributes()

    def __defaultValues(self):
        self.autoPad = "NOTSET"
        self.ceilMode = 0
        self.countIncludePad = 0
        self.dilations = None
        self.kernelShape = None
        self.pads = None
        self.strides = None

    def __initAttributes(self):
        for attr in self._descriptor:
            if attr.name == "auto_pad": # Not tested!
                self.autoPad = attr.s
            elif attr.name == "ceil_mode": # Not tested!
                self.ceilMode = attr.i
            elif attr.name == "count_include_pad": # Not tested!
                self.countIncludePad = attr.i
            elif attr.name == "dilations": # Not tested!
                self.dilations = meta.ONNXIntListAttribute(attr)
            elif attr.name == "kernel_shape":
                self.kernelShape = meta.ONNXIntListAttribute(attr)
            elif attr.name == "pads":
                self.pads = meta.ONNXIntListAttribute(attr)
            elif attr.name == "strides":
                self.strides = meta.ONNXIntListAttribute(attr)
            else:
                err.warning(f"ONNX AveragePool attribute '{attr.name}' is not supported!")
