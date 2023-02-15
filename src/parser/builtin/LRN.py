import onnx.onnx.onnx_ml_pb2 as onnx

import parser.meta.meta as meta

import err

class LRN(meta.ONNXOperatorAttributes):
    alpha: float
    beta: float
    bias: float
    size: int

    def __init__(self, descriptor: list[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)
        self.__defaultValues()
        self.__initAttributes()

    def __defaultValues(self):
        self.alpha = 0.0001
        self.beta = 0.75
        self.bias = 1.0
        self.size = None

    def __initAttributes(self):
        for attr in self._descriptor:
            match attr.name:
                case "alpha":
                    self.alpha = attr.f
                case "beta":
                    self.beta = attr.f
                case "bias":
                    self.bias = attr.f
                case "size":
                    self.size = attr.i
                case _:
                    err.wprint(f"ONNX LRN attribute '{attr.name}' is not supported!")
