import flatbuffers as fb

import lib.tflite.FullyConnectedOptions as fco
import lib.tflite.ActivationFunctionType as aft
import lib.tflite.FullyConnectedOptionsWeightsFormat as wf
import lib.tflite.BuiltinOptions as bo

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
        super().__init__(bo.BuiltinOptions.FullyConnectedOptions)
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
