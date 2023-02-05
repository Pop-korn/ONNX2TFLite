import flatbuffers as fb

import tflite.Buffer as b
import tflite.Model as m

import generator.meta.meta as meta

class Buffer(meta.TFLiteObject):
    data: list

    def __init__(self, data: list=[]) -> None:
        self.data = data

    def genTFLite(self, builder: fb.Builder):
        if len(self.data) == 0:
            # If there is no data, table is empty
            b.Start(builder)
            return b.End(builder)


        b.StartDataVector(builder, len(self.data))

        # IMPORTANT! Flatbuffer is built in reverse, so for correct order,
        # data MUST be iterated in revese
        for ubyte in reversed(self.data): 
            builder.PrependUint8(ubyte) # TODO check data types

        tflData = builder.EndVector()

        b.Start(builder)
        b.AddData(builder, tflData)
        return b.End(builder)


class Buffers(meta.TFLiteVector):
    def __init__(self, vector: list[Buffer]=[]) -> None:
        super().__init__(vector, m.StartBuffersVector)
