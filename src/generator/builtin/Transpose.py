"""
    Transpose

Representation of the TFLite operator 'Transpose'.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

import flatbuffers as fb

import src.generator.meta.meta as meta

import lib.tflite.BuiltinOptions as bOpt
import lib.tflite.BuiltinOperator as bOp
import lib.tflite.TransposeOptions as t

class Transpose(meta.BuiltinOptions):
    def __init__(self) -> None:
        super().__init__(bOpt.BuiltinOptions.TransposeOptions,
                         bOp.BuiltinOperator.TRANSPOSE)
        
    def genTFLite(self, builder: fb.Builder):
        t.Start(builder)
        return t.End(builder)
