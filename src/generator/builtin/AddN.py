"""
    AddN

Representation of the TFLite operator 'AddN'.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

import flatbuffers as fb

import src.generator.meta.meta as meta

from lib.tflite import (
    BuiltinOptions as bOpt,
    BuiltinOperator as bOp,
    AddNOptions as addN
)

class AddN(meta.BuiltinOptions):
    def __init__(self) -> None:
        super().__init__(bOpt.BuiltinOptions.AddNOptions,
                         bOp.BuiltinOperator.ADD_N)

    def genTFLite(self, builder: fb.Builder):
        addN.Start(builder)
        return addN.End(builder)
