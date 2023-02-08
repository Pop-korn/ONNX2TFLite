import flatbuffers as fb

import tflite.SubGraph as sg
import tflite.Tensor as t
import tflite.TensorType as tt

import generator.model.Quantization as Quantization
import generator.meta.meta as meta

""" Classes representing 'Tensor' structure and its Parameters. 'Tensor' is part of the 
    'SubGraph' structure represented in the 'model/SubGraph.py' file.
"""

# TODO If 'hasRank' is false, "shape" must be [].

class Shape(meta.IntVector):
    def __init__(self, shape: list[int]) -> None:
        super().__init__(shape,t.StartShapeVector)

class Tensor(meta.TFLiteObject):
    isVariable: bool
    hasRank: bool
    type: tt.TensorType
    buffer: int
    name: str
    shape: Shape
    quantization: Quantization.Quantization
    # TODO sparsity
    # TODO shapeSignature
    # TODO variantTensors

    def __init__(self, shape: Shape,
                name: str = None, 
                buffer: int = 0, 
                type: tt.TensorType = tt.TensorType.FLOAT32,
                quantization: Quantization.Quantization=None,
                isVariable: bool = False, 
                hasRank: bool = False) -> None:
        self.isVariable = isVariable
        self.hasRank = hasRank
        self.type = type
        self.buffer = buffer
        self.name = name
        self.shape = shape
        self.quantization = quantization

    def genTFLite(self, builder: fb.Builder):
        name = builder.CreateString(self.name)
        tflShape = self.shape.genTFLite(builder)

        if(self.quantization is not None):
            tflQuantization = self.quantization.genTFLite(builder)

        t.Start(builder)

        t.AddType(builder, self.type)
        t.AddIsVariable(builder, self.isVariable)
        t.AddHasRank(builder, self.hasRank)
        t.AddBuffer(builder, self.buffer)
        
        if name is not None:
            t.AddName(builder, name)

        t.AddShape(builder,tflShape)

        if(self.quantization is not None):
            t.AddQuantization(builder,tflQuantization)
        
        return t.End(builder)

class Tensors(meta.TFLiteVector):
    def __init__(self, tensors: list[Tensor] = []) -> None:
        super().__init__(tensors,sg.StartTensorsVector)
