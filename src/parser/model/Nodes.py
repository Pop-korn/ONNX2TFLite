import lib.onnx.onnx.onnx_ml_pb2 as onnx

from typing import List
import src.err as err

import src.parser.meta.meta as meta

from src.parser.builtin import (
    Conv, Dropout, Gemm, LRN, MaxPool, Relu, Reshape, Softmax, 
    BatchNormalization, LeakyRelu
)

class Node(meta.ONNXObject):
    inputs: List[str]
    outputs: List[str]
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
        """ Assign the exact ATTRIBUTES based on the 'opType'. Each operator is represented
            by a unique class in the '/builtin/' directory. """
        match self.opType:
            case "Add":
                self.attributes = None
            case "BatchNormalization":
                self.attributes = BatchNormalization.BatchNormalization(self._descriptor.attribute)
            case "Conv":
                self.attributes = Conv.Conv(self._descriptor.attribute)
            case "Dropout":
                self.attributes = Dropout.Dropout(self._descriptor.attribute)
            case "Gemm":
                self.attributes = Gemm.Gemm(self._descriptor.attribute)
            case "LeakyRelu":
                self.attributes = LeakyRelu.LeakyRelu(self._descriptor.attribute)
            case "LRN":
                self.attributes = LRN.LRN(self._descriptor.attribute)
            case "MaxPool":
                self.attributes = MaxPool.MaxPool(self._descriptor.attribute)
            case "Mul":
                self.attributes = None
            case "Relu":
                self.attributes = Relu.Relu(self._descriptor.attribute)
            case "Reshape":
                self.attributes = Reshape.Reshape(self._descriptor.attribute)
            case "Softmax":
                self.attributes = Softmax.Softmax(self._descriptor.attribute)
            case "Sum":
                self.attributes = None
            case _:
                err.warning(err.Code.UNSUPPORTED_OPERATOR,f"ONNX operator '{self.opType}' is not yet supported!")
            

class Nodes(List[Node]):
    def __init__(self, descriptor: List[onnx.NodeProto]):
        for item in descriptor:
            self.append(Node(item))