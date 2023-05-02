"""
    Graph

Representation of an ONNX 'Graph' object.
Initialized from a protobuf descriptor.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

from typing import List
import lib.onnx.onnx.onnx_ml_pb2 as onnx

import src.parser.model.Nodes as n
import src.parser.model.Tensors as t
import src.parser.model.ValueInfo as vi

import src.parser.meta.meta as meta

class Graph(meta.ONNXObject):
    nodes: n.Nodes
    name: str
    initializers: t.Tensors
    # TODO sparseInitializers
    docString: str
    inputs: List[vi.ValueInfo]
    outputs: List[vi.ValueInfo]
    valueInfo: List[vi.ValueInfo]
    # TODO quantizationAnnotation

    def __init__(self, descriptor: onnx.GraphProto) -> None:
        super().__init__(descriptor)
        self.name = descriptor.name
        self.nodes = n.Nodes(descriptor.node)
        self.initializers = t.Tensors(descriptor.initializer)
        self.docString = descriptor.doc_string
        self.inputs = []
        self.__initList(self.inputs, descriptor.input, vi.ValueInfo)
        self.outputs = []
        self.__initList(self.outputs, descriptor.output, vi.ValueInfo)
        self.valueInfo = []
        self.__initList(self.valueInfo, descriptor.value_info, vi.ValueInfo)

    def __initList(self, lst: List, descriptor, object):
        for item in descriptor:
            lst.append(object(item))