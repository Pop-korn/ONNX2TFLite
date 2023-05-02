"""
    Softmax

Representation of the TFLite operator 'Softmax'.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

import flatbuffers as fb

import lib.tflite.SoftmaxOptions as so
import lib.tflite.BuiltinOptions as bOpt
import lib.tflite.BuiltinOperator as bOp

import src.generator.meta.meta as meta

class Softmax(meta.BuiltinOptions):
    beta: float

    def __init__(self, beta: float) -> None:
        super().__init__(bOpt.BuiltinOptions.SoftmaxOptions,
                         bOp.BuiltinOperator.SOFTMAX)
        self.beta = beta

    def genTFLite(self, builder: fb.Builder):
        so.Start(builder)

        so.AddBeta(builder,self.beta)

        return so.End(builder)