import flatbuffers as fb

import tflite.QuantizationParameters as qp

import generator.meta.meta as meta

""" Classes representing 'Quantization' structure and its Parameters. 'Quantization' is part 
    of the 'Tensor' structure, which is represented in the 'model/Tensor.py' file.
"""

class Min (meta.FloatVector):
    def __init__(self, min: list[float]) -> None:
        super().__init__(min, qp.StartMinVector)

class Max (meta.FloatVector):
    def __init__(self, max: list[float]) -> None:
        super().__init__(max,qp.StartMaxVector)

class Scale(meta.FloatVector):
    def __init__(self, scale: list[float]) -> None:
        super().__init__(scale,qp.StartScaleVector)

class ZeroPoint(meta.IntVector):
    def __init__(self, zeroPoint: list[int]) -> None:
        super().__init__(zeroPoint,qp.StartZeroPointVector,
                            lambda builder : builder.PrependInt64)

class Quantisation:
    min: Min
    max: Max
    scale: Scale
    zeroPoint: ZeroPoint
    quantizedDimension: int

    def __init__(self, min: Min, max: Max, scale: Scale, zeroPoint: ZeroPoint
                , quantizedDimension: int = 0) -> None:
        self.min = min
        self.max = max
        self.scale = scale
        self.zeroPoint = zeroPoint
        self.quantizedDimension = quantizedDimension

    def genTFLite(self, builder: fb.Builder):
        tflMin = self.min.genTFLite(builder)
        tflMax = self.max.genTFLite(builder)
        tflScale = self.scale.genTFLite(builder)
        tflZeroPoint = self.zeroPoint.genTFLite(builder)
        
        qp.Start(builder)

        qp.AddMin(builder, tflMin)
        qp.AddMax(builder, tflMax)
        qp.AddScale(builder, tflScale)
        qp.AddZeroPoint(builder, tflZeroPoint)
        qp.AddQuantizedDimension(builder,self.quantizedDimension)

        return qp.End(builder)