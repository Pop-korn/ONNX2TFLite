import flatbuffers as fb
import numpy as np

import lib.tflite.Buffer as b
import lib.tflite.Model as m
import lib.tflite.TensorType as tt

import src.generator.meta.meta as meta
import src.generator.meta.types as types

class Buffer(meta.TFLiteObject):
    data: list
    type: tt.TensorType

    def __init__(self, data: list=None, 
                type: tt.TensorType=tt.TensorType.INT32) -> None:
        if data is None:
            data = []
        self.data = data
        self.type = type

    def getPrependFunction(self, builder: fb.Builder):
        return types.PrependFunction(builder, self.type)

    def genTFLite(self, builder: fb.Builder):
        if len(self.data) == 0:
            # If there is no data, table is empty
            b.Start(builder)
            return b.End(builder)

        # TODO MASSIVE CHECK NEEDED
        if types.TypeSize(self.type) == 1:
            array = np.array(self.data,np.uint8)
            tflData = builder.CreateNumpyVector(array)
        else:
            PrependFunction = self.getPrependFunction(builder)

            """ 'data' length has to be multiplied by item size, because tflite.Buffer is
                a vector of 'UBYTE's. So e.g. one 'INT32' item will take up 4 spaces in the vector. """
            lenBytes = len(self.data) * types.TypeSize(self.type)
            b.StartDataVector(builder, lenBytes)

            """ IMPORTANT! Flatbuffer is built in reverse, so for correct order,
                data MUST be iterated in revese. """
            for val in reversed(self.data): 
                PrependFunction(val)

            tflData = builder.EndVector()


        b.Start(builder)
        b.AddData(builder, tflData)
        return b.End(builder)


class Buffers(meta.TFLiteVector):
    def __init__(self, vector: list[Buffer]=[]) -> None:
        super().__init__(vector, m.StartBuffersVector)
