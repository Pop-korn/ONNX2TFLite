import flatbuffers as fb

import src.generator.meta.meta as meta

from lib.tflite import (
    BuiltinOptions as bOpt,
    BuiltinOperator as bOp,
    ActivationFunctionType as aft,
    DivOptions as div
)

class Div(meta.BuiltinOptions):
    fusedActivationFunction: aft.ActivationFunctionType

    def __init__(self, 
                fusedActivationFunction: aft.ActivationFunctionType = aft.ActivationFunctionType.NONE
                ) -> None:
        super().__init__(bOpt.BuiltinOptions.DivOptions,
                         bOp.BuiltinOperator.DIV)
        self.fusedActivationFunction = fusedActivationFunction

    def genTFLite(self, builder: fb.Builder):
        div.Start(builder)

        div.AddFusedActivationFunction(builder,self.fusedActivationFunction)

        return div.End(builder)
