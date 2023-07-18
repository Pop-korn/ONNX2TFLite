"""
    Nodes

Representation of an ONNX 'Nodes' object.
Initialized from a protobuf descriptor.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

import lib.onnx.onnx.onnx_ml_pb2 as onnx

from typing import List
import src.err as err

import src.parser.meta.meta as meta

from src.parser.builtin import (
    Conv, Dropout, Gemm, LRN, MaxPool, Relu, Reshape, Softmax, MatMul, Constant,
    BatchNormalization, LeakyRelu, Pad, AveragePool, Transpose, LogSoftmax
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
        if self.opType == "Add":
            self.attributes = None


        elif self.opType == "AveragePool":
            self.attributes = AveragePool.AveragePool(self._descriptor.attribute)
        elif self.opType ==  "BatchNormalization":
            self.attributes = BatchNormalization.BatchNormalization(self._descriptor.attribute)
        elif self.opType ==  "Constant":
            self.attributes = Constant.Constant(self._descriptor.attribute)
        elif self.opType ==  "Conv":
            self.attributes = Conv.Conv(self._descriptor.attribute)
        elif self.opType ==  "Dropout":
            self.attributes = Dropout.Dropout(self._descriptor.attribute)
        elif self.opType ==  "Gemm":
            self.attributes = Gemm.Gemm(self._descriptor.attribute)
        elif self.opType ==  "LeakyRelu":
            self.attributes = LeakyRelu.LeakyRelu(self._descriptor.attribute)
        elif self.opType ==  "LogSoftmax":
            self.attributes = LogSoftmax.LogSoftmax(self._descriptor.attribute)
        elif self.opType ==  "LRN":
            self.attributes = LRN.LRN(self._descriptor.attribute)
        elif self.opType ==  "MatMul":
            self.attributes = MatMul.MatMul(self._descriptor.attribute)
        elif self.opType ==  "MaxPool":
            self.attributes = MaxPool.MaxPool(self._descriptor.attribute)
        elif self.opType ==  "Mul":
            self.attributes = None
        elif self.opType ==  "Pad":
            self.attributes = Pad.Pad(self._descriptor.attribute)
        elif self.opType ==  "Relu":
            self.attributes = Relu.Relu(self._descriptor.attribute)
        elif self.opType ==  "Reshape":
            self.attributes = Reshape.Reshape(self._descriptor.attribute)
        elif self.opType ==  "Softmax":
            self.attributes = Softmax.Softmax(self._descriptor.attribute)
        elif self.opType ==  "Sum":
            self.attributes = None
        elif self.opType ==  "Transpose":
            self.attributes = Transpose.Transpose(self._descriptor.attribute)
        else:
            err.error(err.Code.UNSUPPORTED_OPERATOR,f"ONNX operator '{self.opType}' is not yet supported!")
            

class Nodes(List[Node]):
    def __init__(self, descriptor: List[onnx.NodeProto]):
        for item in descriptor:
            self.append(Node(item))