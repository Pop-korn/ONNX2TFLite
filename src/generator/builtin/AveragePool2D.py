"""
    AveragePool2D

Representation of the TFLite operator 'AveragePool2D'.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

import flatbuffers as fb

import lib.tflite.Pool2DOptions as pool
import lib.tflite.BuiltinOptions as bOpt
import lib.tflite.BuiltinOperator as bOp
import lib.tflite.Padding as p
import lib.tflite.ActivationFunctionType as aft

import src.generator.meta.meta as meta

class AveragePool2D(meta.BuiltinOptions):
    padding: p.Padding
    strideW: int = 0
    strideH: int = 0
    filterW: int = 1
    filterH: int = 1
    fusedActivationFunction: aft.ActivationFunctionType

    def __init__(self, padding: p.Padding = p.Padding.SAME,
                strideW: int = 0, strideH: int = 0,
                filterW: int = 1, filterH: int = 1,
                fusedActivationFunction: aft.ActivationFunctionType = aft.ActivationFunctionType.NONE) -> None:
        super().__init__(bOpt.BuiltinOptions.Pool2DOptions,
                         bOp.BuiltinOperator.AVERAGE_POOL_2D)
        self.padding = padding
        self.strideW = strideW
        self.strideH = strideH
        self.filterW = filterW
        self.filterH = filterH
        self.fusedActivationFunction = fusedActivationFunction

    def genTFLite(self, builder: fb.Builder):
        pool.Start(builder)

        pool.AddPadding(builder, self.padding)
        pool.AddStrideW(builder,self.strideW)
        pool.AddStrideH(builder,self.strideH)
        pool.AddFilterHeight(builder,self.filterH)
        pool.AddFilterWidth(builder,self.filterW)
        pool.AddFusedActivationFunction(builder,self.fusedActivationFunction)

        return pool.End(builder)
    