import onnx.onnx.onnx_ml_pb2 as onnx

import parser.meta.meta as meta

class Conv(meta.ONNXOperatorAttributes):
    # Attribute is 'None' if not present in the model
    autoPad: str
    dilations: meta.ONNXIntListAttribute
    group: int
    kernelShape :meta.ONNXIntListAttribute
    pads: meta.ONNXIntListAttribute
    strides: meta.ONNXIntListAttribute

    def __init__(self, descriptor: list[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)
        self.__defaultValues()
        self.__initAttributes()

    def __defaultValues(self):
        self.autoPad = "NOTSET"
        self.dilations = None
        self.group = 1
        self.kernelShape = None
        self.pads = None
        self.strides = None

    def __initAttributes(self):
        for attr in self._descriptor:
            match attr.name:
                case "auto_pad":
                    self.autoPad = attr.s # Not tested!
                case "dilations":
                    self.dilations = meta.ONNXIntListAttribute(attr) # Not tested!
                case "group":
                    self.group = attr.i 
                case "kernel_shape":
                    self.kernelShape = meta.ONNXIntListAttribute(attr)
                case "pads":
                    self.pads = meta.ONNXIntListAttribute(attr)
                case "strides":
                    self.strides = meta.ONNXIntListAttribute(attr)
