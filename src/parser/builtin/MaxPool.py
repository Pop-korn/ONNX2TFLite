import onnx.onnx.onnx_ml_pb2 as onnx

import parser.meta.meta as meta

class MaxPool(meta.ONNXOperatorAttributes):
    # Attribute is 'None' if not present in the model
    autoPad: str
    ceilMode: int
    dilations: meta.ONNXIntListAttribute
    kernelShape :meta.ONNXIntListAttribute
    pads: meta.ONNXIntListAttribute
    storageOrder: int
    strides: meta.ONNXIntListAttribute

    def __init__(self, descriptor: list[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)
        self.__defaultValues()
        self.__initAttributes()

    def __defaultValues(self):
        self.autoPad = "NOTSET"
        self.ceilMode = 0
        self.dilations = None
        self.kernelShape = None
        self.pads = None
        self.storageOrder = 0
        self.strides = None

    def __initAttributes(self):
        for attr in self._descriptor:
            match attr.name:
                case "auto_pad": # Not tested!
                    self.autoPad = attr.s
                case "ceil_mode": # Not tested!
                    self.ceilMode = attr.i
                case "dilations": # Not tested!
                    self.dilations = meta.ONNXIntListAttribute(attr)
                case "kernel_shape":
                    self.kernelShape = meta.ONNXIntListAttribute(attr)
                case "pads":
                    self.pads = meta.ONNXIntListAttribute(attr)
                case "storage_order": # Not tested!
                    self.storageOrder = attr.i 
                case "strides":
                    self.strides = meta.ONNXIntListAttribute(attr)