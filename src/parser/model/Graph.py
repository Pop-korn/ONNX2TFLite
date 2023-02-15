import onnx.onnx.onnx_ml_pb2 as onnx

import parser.model.Node as n
import parser.model.Tensor as t

import parser.meta.meta as meta

class Graph(meta.ONNXObject):
    # Graph attributes
    nodes: n.Nodes
    name: str
    initializer: t.Tensors
    # TODO sparseInitialize
    docString: str
    # TODO input
    # TODO output
    # TODO valueInfo
    # TODO quantizationAnnotation

    def __init__(self, descriptor: onnx.GraphProto) -> None:
        super().__init__(descriptor)
        self.name = descriptor.name
        self.nodes = n.Nodes(descriptor.node)
        self.initializer = t.Tensors(descriptor.initializer)

        self.docString = descriptor.doc_string