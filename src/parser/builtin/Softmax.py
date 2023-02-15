import onnx.onnx.onnx_ml_pb2 as onnx

import parser.meta.meta as meta

import err

class Softmax(meta.ONNXOperatorAttributes):
    # Attribute is 'None' if not present in the model
    axis: int

    def __init__(self, descriptor: list[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _defaultValues(self):
        self.axis = -1
            
    def _initAttributes(self):
        for attr in self._descriptor:
            match attr.name:
                case "axis": # Not tested!
                    self.axis = attr.i 
                case _:
                    err.wprint(f"ONNX Softmax attribute '{attr.name}' is not supported!")