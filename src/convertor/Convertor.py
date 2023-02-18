import src.generator.model.Tensors as tflT

import src.parser.model.TensorShape as onnxTS

def convertShape(oShape: onnxTS.TensorShape, keepDims: bool = False) -> tflT.Shape:
    dims = [ dim.value for dim in oShape.dims]
    return tflT.Shape(dims)