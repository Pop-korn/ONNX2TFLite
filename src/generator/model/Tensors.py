"""
    Tensors

Module contains classes that represent TFLite 'Tensor' objects.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

from typing import List
import flatbuffers as fb

import lib.tflite.SubGraph as sg
import lib.tflite.Tensor as t
import lib.tflite.TensorType as tt

import src.generator.model.Quantization as Quantization
import src.generator.model.Buffers as Buffers
import src.generator.meta.meta as meta

import src.err as err

""" Classes representing 'Tensor' structure and its Parameters. 'Tensor' is part of the 
    'SubGraph' structure represented in the 'model/SubGraph.py' file.
"""

# TODO If 'hasRank' is false, "shape" must be [].

class Shape(meta.IntVector):
    __shapeOffset: int
    
    __alsoSignature: bool
    __shapeSignatureVector: List[int]
    __shapeSignatureOffset: int

    def __init__(self, shape: List[int]) -> None:
        super().__init__(shape,t.StartShapeVector)
        self.__alsoSignature = False

    def __checkDims(self):
        """ Check if all dimensions are integers. If not, transform this
            to 'shape_signature'. """
        
        self.__shapeSignatureVector = []

        for val in self.vector:
            if type(val) != type(1):
                val = -1
                self.__alsoSignature = True
            
            self.__shapeSignatureVector.append(val)

        if self.__alsoSignature:
            self.vector = [ abs(val) for val in self.__shapeSignatureVector]


    def genTFLite(self, builder: fb.Builder, tensor):
        """ Generates TFLite code for the Shape """
        self.__checkDims()

        if self.__alsoSignature:
            tensor.hasRank = True


        self.__shapeOffset = super().genTFLite(builder)
        if self.__alsoSignature:
            self.vector = self.__shapeSignatureVector
            self.__shapeSignatureOffset = super().genTFLite(builder)

    
    def addTFLite(self, builder):
        t.AddShape(builder, self.__shapeOffset)

        if self.__alsoSignature:
            t.AddShapeSignature(builder, self.__shapeSignatureOffset)


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


    """ IMPORTANT! The following attributes are used only by 'ModelBuilder' 
        in order to make model creation more eficient. """

    """ Reference to the 'Buffer' object holding this tensors data. 'tmpBuffer' MUST be 
        stored a 'Buffers' object and MUST be referenced using the index 'buffer'.  """
    tmpBuffer: Buffers.Buffer

    """ Index to the 'tensors' vector for this tensor. """
    tmpIndex: int

    """ Counter of how many operators use this tensor. """
    tmpReferenceCount: int


    def __init__(self, shape: Shape = None,
                name: str = None, 
                buffer: int = None, 
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

        if self.shape is not None:
            self.shape.genTFLite(builder, self)

        if self.name is not None:
            name = builder.CreateString(self.name)

        if(self.quantization is not None):
            err.requireType(self.quantization, Quantization.Quantization, "Tensor.quantization")
            tflQuantization = self.quantization.genTFLite(builder)

        t.Start(builder)

        if self.shape is not None:
            self.shape.addTFLite(builder)

        t.AddType(builder, self.type)

        if self.buffer is not None:
            t.AddBuffer(builder, self.buffer)

        if self.name is not None:
            t.AddName(builder, name)

        if(self.quantization is not None):
            t.AddQuantization(builder,tflQuantization)

        t.AddIsVariable(builder, self.isVariable)

        t.AddHasRank(builder, self.hasRank)
        
        return t.End(builder)

class Tensors(meta.TFLiteVector):
    def __init__(self, tensors: List[Tensor] = None) -> None:
        super().__init__(tensors,sg.StartTensorsVector)
