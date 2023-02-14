import onnx.onnx.onnx_ml_pb2 as onnx

import parser.meta.meta as meta

class KernelShape(meta.ONNXIntListAttribute):
    pass

class Pads(meta.ONNXIntListAttribute):
    pass

class Strides(meta.ONNXIntListAttribute):
    pass

class Conv(meta.OperatorAttributes):
    # Wrapped descriptor
    __conv: list[onnx.AttributeProto]

    # Conv attributes
    # Attribute is 'None' if not present in the model
    autoPad: str
    kernelShape :KernelShape
    pads: Pads
    strides: Strides

    def __init__(self, descriptor: list[onnx.AttributeProto]) -> None:
        self.__conv = descriptor
        self.autoPad = None
        self.kernelShape = None
        self.pads = None
        self.strides = None
        self.__initAttributes()

    def __initAttributes(self):
        for attr in self.__conv:
            match attr.name:
                case "auto_pad":
                    self.autoPad = attr.s # TODO Not tested!
                case "kernel_shape":
                    self.kernelShape = KernelShape(attr)
                case "pads":
                    self.pads = Pads(attr)
                case "strides":
                    self.strides = Strides(attr)
