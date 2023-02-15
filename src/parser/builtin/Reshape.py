import onnx.onnx.onnx_ml_pb2 as onnx

import parser.meta.meta as meta

import err

class Reshape(meta.ONNXOperatorAttributes):
    # Attribute is 'None' if not present in the model
    allowZero: int

    def __init__(self, descriptor: list[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _defaultValues(self):
        self.allowZero = 0
            
    def _initAttributes(self):
        for attr in self._descriptor:
            match attr.name:
                case "allowzero": # Not tested!
                    self.allowZero = attr.i 
                case _:
                    err.wprint(f"ONNX Reshape attribute '{attr.name}' is not supported!")