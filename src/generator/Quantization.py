import flatbuffers as fb

import tflite.QuantizationParameters as qp

class Min:
    min: list[float]

    def __init__(self, min: list[float]) -> None:
        self.min = min

    def genTFLite(self, builder: fb.Builder):
        qp.StartMinVector(builder, len(self.min))

        for val in self.min:
            builder.PrependFloat32(val)

        return builder.EndVector()

class Max:
    max: list[float]

    def __init__(self, max: list[float]) -> None:
        self.max = max

    def genTFLite(self, builder: fb.Builder):
        qp.StartMaxVector(builder, len(self.max))

        for val in self.max:
            builder.PrependFloat32(val)

        return builder.EndVector()

class Scale:
    scale: list[float]

    def __init__(self, scale: list[float]) -> None:
        self.scale = scale

    def genTFLite(self, builder: fb.Builder):
        qp.StartScaleVector(builder, len(self.scale))

        for val in self.scale:
            builder.PrependFloat32(val)

        return builder.EndVector()

class ZeroPoint:
    zeroPoint: list[int]

    def __init__(self, zeroPoint: list[int]) -> None:
        self.zeroPoint = zeroPoint

    def genTFLite(self, builder: fb.Builder):
        qp.StartZeroPointVector(builder, len(self.zeroPoint))

        for val in self.zeroPoint:
            builder.PrependInt64(val)

        return builder.EndVector()

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