from typing import List
import lib.onnx.onnx.onnx_ml_pb2 as onnx

class OperatorSetId:
    domain: str
    version: int

    def __init__(self, descriptor: onnx.OperatorSetIdProto) -> None:
        self.domain = descriptor.domain
        self.version = descriptor.version

class OperatorSetIds(List[OperatorSetId]):
    def __init__(self, vectorDesc: List[onnx.OperatorSetIdProto]) -> None:
        for item in vectorDesc:
            self.append(OperatorSetId(item))