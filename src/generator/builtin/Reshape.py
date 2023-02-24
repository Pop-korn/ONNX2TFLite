import flatbuffers as fb

import src.generator.meta.meta as meta

import lib.tflite.ReshapeOptions as reshape
import lib.tflite.BuiltinOptions as bo

class NewShape(meta.IntVector):
    def __init__(self, newShape: list[int]) -> None:
        super().__init__(newShape, reshape.StartNewShapeVector)

class Reshape(meta.BuiltinOptions):
    newShape: NewShape

    def __init__(self, newShape: list[int]) -> None:
        super().__init__(bo.BuiltinOptions.ReshapeOptions)
        self.newShape = NewShape(newShape)

    def genTFLite(self, builder: fb.Builder):
        tflNewShape = self.newShape.genTFLite(builder)

        reshape.Start(builder)

        reshape.AddNewShape(builder, tflNewShape)

        return reshape.End(builder)