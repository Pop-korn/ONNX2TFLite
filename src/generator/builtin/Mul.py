"""
    Mul

Representation of the TFLite operator 'Mul'.

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
    MulOptions as mul
)


class Mul(meta.BuiltinOptions):
    fusedActivationFunction: aft.ActivationFunctionType

    def __init__(self, 
                fusedActivationFunction: aft.ActivationFunctionType = aft.ActivationFunctionType.NONE
                ) -> None:
        super().__init__(bOpt.BuiltinOptions.MulOptions,
                         bOp.BuiltinOperator.MUL)
        self.fusedActivationFunction = fusedActivationFunction

    def genTFLite(self, builder: fb.Builder):
        mul.Start(builder)

        mul.AddFusedActivationFunction(builder,self.fusedActivationFunction)

        return mul.End(builder)
