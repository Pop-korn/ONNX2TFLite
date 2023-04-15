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
            match attr.name:
                case "auto_pad": # Not tested!
                    self.autoPad = attr.s
                case "ceil_mode": # Not tested!
                    self.ceilMode = attr.i
                case "count_include_pad": # Not tested!
                    self.countIncludePad = attr.i 
                case "dilations": # Not tested!
                    self.dilations = meta.ONNXIntListAttribute(attr)
                case "kernel_shape":
                    self.kernelShape = meta.ONNXIntListAttribute(attr)
                case "pads":
                    self.pads = meta.ONNXIntListAttribute(attr)
                case "strides":
                    self.strides = meta.ONNXIntListAttribute(attr)
                case _:
                    err.warning(f"ONNX MaxPool attribute '{attr.name}' is not supported!")
