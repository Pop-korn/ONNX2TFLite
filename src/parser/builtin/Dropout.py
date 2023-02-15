import onnx.onnx.onnx_ml_pb2 as onnx

import parser.meta.meta as meta

import err

class Dropout(meta.ONNXOperatorAttributes):
    # Attribute is 'None' if not present in the model
    seed: int

    def __init__(self, descriptor: list[onnx.AttributeProto]) -> None:
        super().__init__(descriptor)

    def _defaultValues(self):
        self.seed = None

    def _initAttributes(self):
        for attr in self._descriptor:
            match attr.name:
                case "seed": # Not tested!
                    self.seed = attr.i
                case _:
                    err.wprint(f"ONNX Dropout attribute '{attr.name}' is not supported!")
