import onnx.onnx.onnx_ml_pb2 as onnx

import parser.meta.meta as meta

class Pads(list [int]):
    __pads: onnx.AttributeProto

    name: str
    type: meta.AttributeType

    def __init__(self, descriptor: onnx.AttributeProto):
        self.__pads = descriptor

        self.name = descriptor.name
        self.type = descriptor.type

        for item in descriptor.ints:
            self.append(item)


class Conv(meta.OperatorAttributes):
    # Wrapped descriptor
    __conv: list[onnx.AttributeProto]

    # Conv attributes
    # Attribute is 'None' if not present in the model
    autoPad: str
    pads: Pads

    def __init__(self, descriptor: list[onnx.AttributeProto]) -> None:
        self.__conv = descriptor
        self.autoPad = None
        self.__initAttributes()

    def __initAttributes(self):
        for attr in self.__conv:
            match attr.name:
                case "auto_pad":
                    self.autoPad = attr.s
                case "pads":
                    self.pads = Pads(attr)
