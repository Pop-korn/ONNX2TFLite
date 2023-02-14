import onnx.onnx.onnx_ml_pb2 as onnx

import parser.model.Node as n

import parser.meta.meta as meta

class Graph(meta.ONNXObject):
    # Graph attributes
    nodes: n.Nodes
    name: str

    def __init__(self, descriptor: onnx.GraphProto) -> None:
        super().__init__(descriptor)
        self.name = descriptor.name
        self.nodes = n.Nodes(descriptor.node)