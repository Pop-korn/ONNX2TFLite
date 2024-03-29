"""
    Quantization

Module contains classes that represent TFLite 'Quantization' objects.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

from typing import List
import flatbuffers as fb

import lib.tflite.QuantizationParameters as qp
import lib.tflite.QuantizationDetails as qd

import src.generator.meta.meta as meta

""" Classes representing 'Quantization' structure and its Parameters. 'Quantization' is part 
    of the 'Tensor' structure, which is represented in the 'model/Tensor.py' file.
"""

class Min (meta.FloatVector):
    def __init__(self, min: List[float] = None) -> None:
        super().__init__(min, qp.StartMinVector, genEmpty = False)

class Max (meta.FloatVector):
    def __init__(self, max: List[float] = None) -> None:
        super().__init__(max,qp.StartMaxVector, genEmpty = False)

class Scale(meta.FloatVector):
    def __init__(self, scale: List[float] = None) -> None:
        super().__init__(scale,qp.StartScaleVector)

class ZeroPoint(meta.IntVector):
    def __init__(self, zeroPoint: List[int] = None) -> None:
        super().__init__(zeroPoint,qp.StartZeroPointVector,
                        lambda builder : builder.PrependInt64)

class Quantization(meta.TFLiteObject):
    min: Min
    max: Max
    scale: Scale
    zeroPoint: ZeroPoint
    quantizedDimension: int
    detailsType: qd.QuantizationDetails
    # TODO details

    def __init__(self, min: Min=Min(), max: Max=Max(),
                scale: Scale=None, 
                zeroPoint: ZeroPoint=ZeroPoint([0]),
                quantizedDimension: int = 0, 
                detailsType: qd.QuantizationDetails=qd.QuantizationDetails.NONE) -> None:
        self.min = min
        self.max = max
        self.scale = scale
        self.zeroPoint = zeroPoint
        self.quantizedDimension = quantizedDimension
        self.detailsType = detailsType

    def genTFLite(self, builder: fb.Builder):
        tflMin = self.min.genTFLite(builder)
        tflMax = self.max.genTFLite(builder)
        tflScale = self.scale.genTFLite(builder)
        tflZeroPoint = self.zeroPoint.genTFLite(builder)
        



        qp.Start(builder)

        if tflMin is not None:
            qp.AddMin(builder, tflMin)

        if tflMax is not None:
            qp.AddMax(builder, tflMax)

        qp.AddScale(builder, tflScale)

        qp.AddZeroPoint(builder, tflZeroPoint)

        qp.AddDetailsType(builder, self.detailsType)
        
        qp.AddQuantizedDimension(builder,self.quantizedDimension)

        return qp.End(builder)
        