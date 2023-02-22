import flatbuffers as fb

import src.generator.meta.meta as meta

import lib.tflite.BuiltinOptions as bo
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
        super().__init__(bo.BuiltinOptions.LocalResponseNormalizationOptions)
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
