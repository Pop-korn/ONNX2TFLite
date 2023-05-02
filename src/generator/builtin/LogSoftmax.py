"""
    LogSoftmax

Representation of the TFLite operator 'LogSoftmax'.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

import flatbuffers as fb

import lib.tflite.LogSoftmaxOptions as lso
import lib.tflite.BuiltinOptions as bOpt
import lib.tflite.BuiltinOperator as bOp

import src.generator.meta.meta as meta

class LogSoftmax(meta.BuiltinOptions):
    def __init__(self) -> None:
        super().__init__(bOpt.BuiltinOptions.LogSoftmaxOptions,
                         bOp.BuiltinOperator.LOG_SOFTMAX)

    def genTFLite(self, builder: fb.Builder):
        lso.Start(builder)
        return lso.End(builder)