"""
    LRN

Representation of the TFLite operator 'LocalResponseNormalization'.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

import flatbuffers as fb

import src.generator.meta.meta as meta

import lib.tflite.BuiltinOptions as bOpt
import lib.tflite.BuiltinOperator as bOp
import lib.tflite.LocalResponseNormalizationOptions as lrn

class LRN(meta.BuiltinOptions):
    radius: int
    bias: float
    alpha: float
    beta: float

    def __init__(self, radius: int = 0,
                bias: float = 0.0,
                alpha: float = 0.0,
                beta: float = 0.0) -> None:
        super().__init__(bOpt.BuiltinOptions.LocalResponseNormalizationOptions,
                         bOp.BuiltinOperator.LOCAL_RESPONSE_NORMALIZATION)
        self.radius = radius
        self.bias = bias
        self.alpha = alpha
        self.beta = beta

    def genTFLite(self, builder: fb.Builder):
        lrn.Start(builder)

        lrn.AddRadius(builder, self.radius)
        lrn.AddBias(builder, self.bias)
        lrn.AddAlpha(builder, self.alpha)
        lrn.AddBeta(builder, self.beta)

        return lrn.End(builder)
