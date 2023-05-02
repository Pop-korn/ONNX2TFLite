"""
    Sub

Representation of the TFLite operator 'Sub'.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

import flatbuffers as fb

import src.generator.meta.meta as meta

from lib.tflite import (
    BuiltinOptions as bOpt,
    BuiltinOperator as bOp,
    ActivationFunctionType as aft,
    SubOptions as sub
)


class Sub(meta.BuiltinOptions):
    fusedActivationFunction: aft.ActivationFunctionType
    # TODO potScaleInt16

    def __init__(self, 
                fusedActivationFunction: aft.ActivationFunctionType = aft.ActivationFunctionType.NONE
                ) -> None:
        super().__init__(bOpt.BuiltinOptions.SubOptions,
                         bOp.BuiltinOperator.SUB)
        self.fusedActivationFunction = fusedActivationFunction

    def genTFLite(self, builder: fb.Builder):
        sub.Start(builder)

        sub.AddFusedActivationFunction(builder,self.fusedActivationFunction)

        return sub.End(builder)
