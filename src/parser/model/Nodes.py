import lib.onnx.onnx.onnx_ml_pb2 as onnx

import src.err as err

import src.parser.meta.meta as meta

import src.parser.builtin.Conv as Conv
import src.parser.builtin.Dropout as Dropout
import src.parser.builtin.Gemm as Gemm
import src.parser.builtin.LRN as LRN
import src.parser.builtin.MaxPool as MaxPool
import src.parser.builtin.Relu as Relu
import src.parser.builtin.Reshape as Reshape
import src.parser.builtin.Softmax as Softmax

class Node(meta.ONNXObject):
    inputs: list[str]
    outputs: list[str]
    name: str
    opType: str
    domain: str
    attributes: meta.ONNXOperatorAttributes
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
                self.attributes = Conv.Conv(self._descriptor.attribute)
            case "Dropout":
                self.attributes = Dropout.Dropout(self._descriptor.attribute)
            case "Gemm":
                self.attributes = Gemm.Gemm(self._descriptor.attribute)
            case "LRN":
                self.attributes = LRN.LRN(self._descriptor.attribute)
            case "MaxPool":
                self.attributes = MaxPool.MaxPool(self._descriptor.attribute)
            case "Relu":
                self.attributes = Relu.Relu(self._descriptor.attribute)
            case "Reshape":
                self.attributes = Reshape.Reshape(self._descriptor.attribute)
            case "Softmax":
                self.attributes = Softmax.Softmax(self._descriptor.attribute)
            case _:
                err.warning(err.Code.UNSUPPORTED_OPERATOR,f"ONNX operator '{self.opType}' is not yet supported!")
            

class Nodes(list[Node]):
    def __init__(self, descriptor: list[onnx.NodeProto]):
        for item in descriptor:
            self.append(Node(item))