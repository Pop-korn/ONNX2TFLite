import flatbuffers as fb

import tflite.SubGraph as sg
import tflite.Tensor as t
import tflite.TensorType as tt

import generator.model.Quantization as Quantization
import generator.meta.meta as meta

""" Classes representing 'Tensor' structure and its Parameters. 'Tensor' is part of the 
    'SubGraph' structure represented in the 'model/SubGraph.py' file.
"""

class Shape(meta.IntVector):
    def __init__(self, shape: list[int]) -> None:
        super().__init__(shape,t.StartShapeVector)

class Tensor:
    isVariable: bool
    hasRank: bool
    type: tt.TensorType
    buffer: int
    name: str
    shape: Shape
    quantization: Quantization.Quantisation
    # TODO sparsity
    # TODO shapeSignature
    # TODO variantTensors

    def __init__(self, quantization: Quantization.Quantisation, shape: Shape,
     name: str = None, buffer: int = 0, type: tt.TensorType = tt.TensorType.FLOAT32,
     isVariable: bool = False, hasRank: bool = False) -> None:
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
        tflQuantization = self.quantization.genTFLite(builder)

        t.Start(builder)

        t.AddType(builder, self.type)
        t.AddIsVariable(builder, self.isVariable)
        t.AddHasRank(builder, self.hasRank)
        t.AddBuffer(builder, self.buffer)
        if name is not None:
            t.AddName(builder, name)

        t.AddShape(builder,tflShape)
        t.AddQuantization(builder,tflQuantization)
        
        return t.End(builder)

class Tensors:
    tensors: list[Tensor]

    def __init__(self, tensors: list[Tensor]) -> None:
        self.tensors = tensors

    def genTFLite(self, builder: fb.Builder):
        tflTensors = []

        for tensor in self.tensors:
            tflTensors.append(tensor.genTFLite(builder))

        sg.StartTensorsVector(builder, len(self.tensors))

        for tflTensor in tflTensors:
            builder.PrependSOffsetTRelative(tflTensor)

        return builder.EndVector()



    
