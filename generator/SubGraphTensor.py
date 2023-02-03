import flatbuffers as fb
import tflite.SubGraph as sg
import tflite.Tensor as t

class Tensor:
    isVariable: bool

    def __init__(self, isVariable: bool) -> None:
        self.isVariable = isVariable

    def genTFLite(self, builder: fb.Builder):
        t.Start(builder)

        t.AddIsVariable(builder, self.isVariable)
        
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



    
