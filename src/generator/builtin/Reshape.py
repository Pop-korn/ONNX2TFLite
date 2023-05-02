"""
    Reshape

Representation of the TFLite operator 'Reshape'.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

from typing import List
import flatbuffers as fb

import src.generator.meta.meta as meta

import lib.tflite.ReshapeOptions as reshape
import lib.tflite.BuiltinOptions as bOpt
import lib.tflite.BuiltinOperator as bOp

class NewShape(meta.IntVector):
    def __init__(self, newShape: List[int]) -> None:
        super().__init__(newShape, reshape.StartNewShapeVector)

class Reshape(meta.BuiltinOptions):
    newShape: NewShape

    def __init__(self, newShape: List[int]) -> None:
        super().__init__(bOpt.BuiltinOptions.ReshapeOptions,
                         bOp.BuiltinOperator.RESHAPE)
        self.newShape = NewShape(newShape)

    def genTFLite(self, builder: fb.Builder):
        tflNewShape = self.newShape.genTFLite(builder)

        reshape.Start(builder)

        reshape.AddNewShape(builder, tflNewShape)

        return reshape.End(builder)
