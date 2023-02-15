import onnx.onnx.onnx_ml_pb2 as onnx

import err

import parser.meta.meta as meta

import parser.builtin.Conv as conv
import parser.builtin.LRN as lrn
import parser.builtin.MaxPool as mp
import parser.builtin.Relu as relu
import parser.builtin.Reshape as r

class Node(meta.ONNXObject):
    inputs: list[str]
    outputs: list[str]
    name: str
    opType: str
    attributes: meta.ONNXOperatorAttributes
    domain: str
    docString: str

    def __init__(self, descriptor: onnx.NodeProto) -> None:
        super().__init__(descriptor)
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
                self.attributes = conv.Conv(self._descriptor.attribute)
            case "LRN":
                self.attributes = lrn.LRN(self._descriptor.attribute)
            case "MaxPool":
                self.attributes = mp.MaxPool(self._descriptor.attribute)
            case "Relu":
                self.attributes = relu.Relu(self._descriptor.attribute)
            case "Reshape":
                self.attributes = r.Reshape(self._descriptor.attribute)
            case _:
                err.wprint(err.Code.UNSUPPORTED_OPERATOR,f"ONNX operator '{self.opType}' is not yet supported!")
            

class Nodes(list[Node]):
    def __init__(self, descriptor: list[onnx.NodeProto]):
        for item in descriptor:
            self.append(Node(item))
            if self[-1].opType == "":
                print(item)