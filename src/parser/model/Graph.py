import onnx.onnx.onnx_ml_pb2 as onnx

import parser.model.Node as n

class Graph:
    # Wrapped descriptor
    __graph: onnx.GraphProto

    # Graph attributes
    nodes: n.Nodes
    name: str

    def __init__(self, descriptor: onnx.GraphProto) -> None:
        self.__graph = descriptor
        self.name = descriptor.name
        self.nodes = n.Nodes(descriptor.node)