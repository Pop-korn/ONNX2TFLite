from typing import List
import flatbuffers as fb
import numpy as np

import lib.tflite.Buffer as b
import lib.tflite.Model as m
import lib.tflite.TensorType as tt

import src.generator.meta.meta as meta
import src.generator.meta.types as types

import src.err as err

class Buffer(meta.TFLiteObject):
    """ 'data' is an array of any type, but MUST have the correct 'dtype' specified! """
    data: np.ndarray
    type: tt.TensorType


    """ IMPORTANT! The following attributes are used only by 'ModelBuilder' 
        in order to make model creation more eficient. """

    """ Index to the 'buffers' vector. Used to assign the 'buffer' attribute of the 
        Tensor, this buffer belongs to."""
    tmpIndex: int

    """ Indicator if the buffer is used in the final model. """
    tmpUsed: bool


    def __init__(self, data: np.ndarray=None, 
                type: tt.TensorType=tt.TensorType.INT32) -> None:
        self.data = data
        self.type = type

    def __dataIsEmpty(self):
        """ Determine if the buffer data is empty. """
        return (self.data is None) or (len(self.data) == 0)

    # -------------------- OLD IMPLEMENTATION (SLOW) --------------------
    # def getPrependFunction(self, builder: fb.Builder):
    #     return types.PrependFunction(builder, self.type)

    def genTFLite(self, builder: fb.Builder):
        if self.__dataIsEmpty():
            # If there is no data, table is empty
            b.Start(builder)
            return b.End(builder)

        # -------------------- OLD IMPLEMENTATION (SLOW) --------------------
        # PrependFunction = self.getPrependFunction(builder)
        # """ 'data' length has to be multiplied by item size, because tflite.Buffer is
        #     a vector of 'UBYTE's. So e.g. one 'INT32' item will take up 4 spaces in the vector. """
        # lenBytes = len(self.data) * types.TypeSize(self.type)
        # b.StartDataVector(builder, lenBytes)
        # """ IMPORTANT! Flatbuffer is built in reverse, so for correct order,
        #     data MUST be iterated in revese. """
        # for val in reversed(self.data): 
        #     PrependFunction(val)
        # tflData = builder.EndVector()

        err.requireType(self.data,np.ndarray,"Buffer.data")

        if self.data.dtype.itemsize != 1:
            self.data = np.frombuffer(self.data.tobytes(),np.uint8)

        tflData = builder.CreateNumpyVector(self.data)
        # In case of problems, see 'https://github.com/google/flatbuffers/issues/4668'.

        b.Start(builder)
        b.AddData(builder, tflData)

        return b.End(builder)


class Buffers(meta.TFLiteVector):
    def __init__(self, vector: List[Buffer]=[]) -> None:
        super().__init__(vector, m.StartBuffersVector)
