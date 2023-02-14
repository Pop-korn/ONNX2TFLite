import onnx.onnx.onnx_ml_pb2 as onnx

import err

import parser.meta.meta as meta

import parser.builtin.Conv as conv

class Node:
    # Wrapped descriptor
    __node: onnx.NodeProto

    # Node attributes
    inputs: list[str]
    outputs: list[str]
    name: str
    opType: str
    attributes: meta.OperatorAttributes
    domain: str
    docString: str

    def __init__(self, descriptor: onnx.NodeProto) -> None:
        self.__node = descriptor
        self.inputs = descriptor.input
        self.outputs = descriptor.output
        self.name = descriptor.name
        self.opType = descriptor.op_type
        self.domain = descriptor.domain
        self.docString = descriptor.doc_string

        self.__assignAttributes()

    def __assignAttributes(self):
        """ Assign the exact attributes based on the 'opType'. Each operator is represented
            by a unique class in the '/builtin/' directory. """
        match self.opType:
            case "Conv":
                self.attributes = conv.Conv(self.__node.attribute)
            case _:
                err.wprint(err.Code.UNSUPPORTED_OPERATOR,f"ONNX operator '{self.opType}' is not yet supported!")
            

class Nodes(list[Node]):
    def __init__(self, descriptor: list[onnx.NodeProto]):
        for item in descriptor:
            print(item)
            self.append(Node(item))