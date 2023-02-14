import onnx.onnx.onnx_ml_pb2 as onnx

class OperatorAttributes:
    """ Parent class of every class in the '/builtin/' directory.
        Represents an operator with its specific attributes. """
    pass

""" AttributeType enum reference """
AttributeType = onnx.AttributeProto.AttributeType
        