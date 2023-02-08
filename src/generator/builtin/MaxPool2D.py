import flatbuffers as fb

import tflite.Pool2DOptions as pool
import tflite.BuiltinOptions as bo
import tflite.Padding as p
import tflite.ActivationFunctionType as aft

import generator.meta.meta as meta

class MaxPool2D(meta.BuiltinOptions):
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
        super().__init__(bo.BuiltinOptions.Pool2DOptions)
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