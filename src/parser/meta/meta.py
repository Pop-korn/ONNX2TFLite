import onnx.onnx.onnx_ml_pb2 as onnx

class OperatorAttributes:
    """ Parent class of every class in the '/builtin/' directory.
        Represents an operator with its specific attributes. """
    pass

""" AttributeType enum reference """
AttributeType = onnx.AttributeProto.AttributeType

class ONNXIntListAttribute(list [int]):
    """ Represents an ONNX operator attribute, that is a list of integers
        and has 'name' and 'type'.  """
    __descriptor: onnx.AttributeProto

    name: str
    type: AttributeType

    def __init__(self, descriptor: onnx.AttributeProto):
        self.__descriptor = descriptor

        self.name = descriptor.name
        self.type = descriptor.type

        for item in descriptor.ints:
            self.append(item)
        