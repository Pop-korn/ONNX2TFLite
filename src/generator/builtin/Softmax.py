import flatbuffers as fb

import lib.tflite.SoftmaxOptions as so
import lib.tflite.BuiltinOptions as bo

import src.generator.meta.meta as meta

class Softmax(meta.BuiltinOptions):
    beta: float

    def __init__(self, beta: float) -> None:
        super().__init__(bo.BuiltinOptions.SoftmaxOptions)
        self.beta = beta

    def genTFLite(self, builder: fb.Builder):
        so.Start(builder)

        so.AddBeta(builder,self.beta)

        return so.End(builder)