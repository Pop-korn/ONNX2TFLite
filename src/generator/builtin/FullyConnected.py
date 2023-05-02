"""
    FullyConnected

Representation of the TFLite operator 'FullyConnected'.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

import flatbuffers as fb

import lib.tflite.FullyConnectedOptions as fco
import lib.tflite.ActivationFunctionType as aft
import lib.tflite.FullyConnectedOptionsWeightsFormat as wf
import lib.tflite.BuiltinOptions as bOpt
import lib.tflite.BuiltinOperator as bOp

import src.generator.meta.meta as meta

class FullyConnected(meta.BuiltinOptions):
    fusedActivationFunction: aft.ActivationFunctionType
    weightsFormat: wf.FullyConnectedOptionsWeightsFormat
    keepNumDims: bool
    asymmetricQuantizeInputs: bool

    def __init__(self,fusedActivationFunction: aft.ActivationFunctionType=aft.ActivationFunctionType.NONE,
                weightsFormat: wf.FullyConnectedOptionsWeightsFormat=wf.FullyConnectedOptionsWeightsFormat.DEFAULT,
                keepNumDims: bool=False,
                asymmetricQuantizeInputs: bool=False) -> None:
        super().__init__(bOpt.BuiltinOptions.FullyConnectedOptions,
                         bOp.BuiltinOperator.FULLY_CONNECTED)
        self.fusedActivationFunction = fusedActivationFunction
        self.weightsFormat = weightsFormat
        self.keepNumDims = keepNumDims
        self.asymmetricQuantizeInputs = asymmetricQuantizeInputs

    def genTFLite(self, builder: fb.Builder):
        fco.Start(builder)

        fco.AddFusedActivationFunction(builder, self.fusedActivationFunction)
        fco.AddWeightsFormat(builder, self.weightsFormat)
        fco.AddKeepNumDims(builder, self.keepNumDims)
        fco.AddAsymmetricQuantizeInputs(builder, self.asymmetricQuantizeInputs)

        return fco.End(builder)
