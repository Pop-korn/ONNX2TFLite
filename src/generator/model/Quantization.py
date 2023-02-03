import flatbuffers as fb

import tflite.QuantizationParameters as qp

import generator.meta.meta as meta

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
        minTFLite = self.min.genTFLite(builder)
        maxTFLite = self.max.genTFLite(builder)
        scaleTFLite = self.scale.genTFLite(builder)
        zeroPointTFLite = self.zeroPoint.genTFLite(builder)
        
        qp.Start(builder)

        qp.AddMin(builder, minTFLite)
        qp.AddMax(builder, maxTFLite)
        qp.AddScale(builder, scaleTFLite)
        qp.AddZeroPoint(builder, zeroPointTFLite)
        qp.AddQuantizedDimension(builder,self.quantizedDimension)

        return qp.End(builder)