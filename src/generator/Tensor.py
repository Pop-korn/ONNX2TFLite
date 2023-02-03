import flatbuffers as fb

import tflite.SubGraph as sg
import tflite.Tensor as t
import tflite.TensorType as tt

import generator.Quantization as Quantization

class Shape:
    shape: list[int]

    def __init__(self, shape: list[int]) -> None:
        self.shape = shape

    def genTFLite(self, builder: fb.Builder):
        t.StartShapeVector(builder, len(self.shape))

        for dimension in self.shape:
            builder.PrependInt32(dimension)

        return builder.EndVector()


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
        shapeTFLite = self.shape.genTFLite(builder)
        quantizationTFLite = self.quantization.genTFLite(builder)

        t.Start(builder)

        t.AddType(builder, self.type)
        t.AddIsVariable(builder, self.isVariable)
        t.AddHasRank(builder, self.hasRank)
        t.AddBuffer(builder, self.buffer)
        if name is not None:
            t.AddName(builder, name)

        t.AddShape(builder,shapeTFLite)
        t.AddQuantization(builder,quantizationTFLite)
        
        return t.End(builder)

class Tensors:
    tensors: list[Tensor]

    def __init__(self, tensors: list[Tensor]) -> None:
        self.tensors = tensors

    def genTFLite(self, builder: fb.Builder):
        tfliteTensors = []

        for tensor in self.tensors:
            tfliteTensors.append(tensor.genTFLite(builder))

        sg.StartTensorsVector(builder, len(self.tensors))

        for tfliteTensor in tfliteTensors:
            builder.PrependSOffsetTRelative(tfliteTensor)

        return builder.EndVector()



    
