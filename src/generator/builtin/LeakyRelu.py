"""
    LeakyRelu

Representation of the TFLite operator 'LeakyRelu'.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

import flatbuffers as fb

import lib.tflite.LeakyReluOptions as lr
import lib.tflite.BuiltinOptions as bOpt
import lib.tflite.BuiltinOperator as bOp

import src.generator.meta.meta as meta

class LeakyRelu(meta.BuiltinOptions):
    alpha: float

    def __init__(self, alpha: float) -> None:
        super().__init__(bOpt.BuiltinOptions.LeakyReluOptions,
                         bOp.BuiltinOperator.LEAKY_RELU)
        self.alpha = alpha

    def genTFLite(self, builder: fb.Builder):
        lr.Start(builder)

        lr.AddAlpha(builder,self.alpha)

        return lr.End(builder)
    