import lib.onnx.onnx.onnx_ml_pb2 as onnx

""" Various ENUM references. Originals are 'difficult' to acces because of no intellisense. """
AttributeType = onnx.AttributeProto.AttributeType
DataType = onnx.TensorProto.DataType
DataLocation = onnx.TensorProto.DataLocation

class ONNXObject:
    _descriptor: onnx.DESCRIPTOR

    def __init__(self, descriptor: onnx.DESCRIPTOR) -> None:
        self._descriptor = descriptor

class ONNXOperatorAttributes:
    """ Parent class of every class in the '/builtin/' directory.
        Represents an operator with its specific attributes. """

    """ Protobuf descriptor. Holds barely structured data, that represents the individual
        attributes of the operator. The data will be assigned to the subclasses attributes 
        for easier access. """
    _descriptor: list[onnx.AttributeProto]

    def __init__(self, descriptor: onnx.AttributeProto) -> None:
        self._descriptor = descriptor
        self._defaultValues()
        self._initAttributes()

    def _defaultValues(self):
        """ Chlid class should assing default values to its attributes or 'None'
            if it doesn't have a default value. """
        pass

    def _initAttributes(self):
        """ Child class should initialize its attributes with values from the '_descriptor'. """
        pass

class ONNXIntListAttribute(list [int]):
    """ Represents an ONNX operator attribute, that is a list of integers
        and has 'name' and 'type'.  """
    _descriptor: onnx.AttributeProto

    name: str
    type: AttributeType

    def __init__(self, descriptor: onnx.AttributeProto):
        self._descriptor = descriptor

        self.name = descriptor.name
        self.type = descriptor.type

        for item in descriptor.ints:
            self.append(item)

def isDefined(descriptor: onnx.DESCRIPTOR):
    """ Tetermine if given descriptor is not empty. """
    return descriptor != ""
        