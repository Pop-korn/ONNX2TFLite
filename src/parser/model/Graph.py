import onnx.onnx.onnx_ml_pb2 as onnx

class Graph:
    # Wrapped descriptor
    __graph: onnx.GraphProto

    def __init__(self, descriptor: onnx.GraphProto) -> None:
        self.__graph = descriptor