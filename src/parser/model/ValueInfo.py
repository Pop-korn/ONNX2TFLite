import lib.onnx.onnx.onnx_ml_pb2 as onnx

import src.parser.meta.meta as meta
import src.parser.model.Type as t

class ValueInfo(meta.ONNXObject):
    name: str
    type: t.Type
    docString: str

    def __init__(self, descriptor: onnx.ValueInfoProto) -> None:
        super().__init__(descriptor)
        self.name = descriptor.name
        self.type = t.Type(descriptor.type)
        self.docString = descriptor.doc_string
