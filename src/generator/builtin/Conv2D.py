"""
    Conv2D

Representation of the TFLite operator 'Conv2D'.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

import flatbuffers as fb

import src.generator.meta.meta as meta

import lib.tflite.BuiltinOptions as bOpt
import lib.tflite.BuiltinOperator as bOp
import lib.tflite.Conv2DOptions as conv
import lib.tflite.Padding as p
import lib.tflite.ActivationFunctionType as aft

class Conv2D(meta.BuiltinOptions):
    padding: p.Padding
    strideW: int = 0
    strideH: int = 0
    dilationWFactor: int = 1
    dilationHFactor: int = 1
    fusedActivationFunction: aft.ActivationFunctionType

    def __init__(self, padding: p.Padding = p.Padding.SAME,
                strideW: int = 0, strideH: int = 0,
                dilationWFactor: int = 1, dilationHFactor: int = 1,
                fusedActivationFunction: aft.ActivationFunctionType = aft.ActivationFunctionType.NONE) -> None:
        super().__init__(bOpt.BuiltinOptions.Conv2DOptions,
                         bOp.BuiltinOperator.CONV_2D)
        self.padding = padding
        self.strideW = strideW
        self.strideH = strideH
        self.dilationWFactor = dilationWFactor
        self.dilationHFactor = dilationHFactor
        self.fusedActivationFunction = fusedActivationFunction

    def genTFLite(self, builder: fb.Builder):
        conv.Start(builder)

        conv.AddPadding(builder, self.padding)
        conv.AddStrideW(builder,self.strideW)
        conv.AddStrideH(builder,self.strideH)
        conv.AddFusedActivationFunction(builder,self.fusedActivationFunction)
        conv.AddDilationWFactor(builder,self.dilationWFactor)
        conv.AddDilationHFactor(builder,self.dilationHFactor)

        return conv.End(builder)

    