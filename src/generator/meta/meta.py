from typing_extensions import override
import flatbuffers as fb
from typing import Callable

import tflite.BuiltinOptions as bo

import err

""" This file contains parent classes for simple classes used in the '/model' directory. """

class TFLiteObject:
    """ Parent class for all tflite objects. That is all objects in the 'generator' directory. """
    
    """ Generates tflite representation for this object. MUST be overriden! """
    def genTFLite(self, builder: fb.Builder):
        err.eprint("TFLiteObject: genTFLite() is not defined!")

class TFLiteVector(TFLiteObject):
    """ Represents a TFLite vector of TFLiteObjects. Provides interface for storing data
        and generating output TFLite code. """
    vector: list[TFLiteObject]

    """ TFLite 'Start...Vector' function for the exact vector. Takes 2 arguments, 
    'floatbuffers.Builder' and number of vector elements """
    StartFunction: Callable[[fb.Builder, int],None]

    """ TFLite 'Prepend...' function for the exact vector item type. Takes 'flatbuffers.Builder' 
    as argument """
    PrependFunction: Callable[[fb.Builder],None]

    def __init__(self, vector: list[TFLiteObject], StartFunction: Callable[[fb.Builder, int],None]
                , PrependFunction: Callable[[fb.Builder],Callable[[int],None]] = lambda builder: builder.PrependUOffsetTRelative) -> None:
        self.vector = vector
        self.StartFunction = StartFunction
        self.PrependFunction = PrependFunction

    def append(self, item):
        self.vector.append(item)

    def get(self, index: int):
        return self.vector[index]

    def genTFLite(self, builder: fb.Builder):
        """ Generates TFLite code for the vector """
        # IMPORTANT! tflite MUST be generated for list items in REVERSE ORDER! 
        # Otherwise the order will be wrong.
        tflVector = [item.genTFLite(builder) for item in reversed(self.vector)]

        self.StartFunction(builder, len(self.vector))

        for tflItem in tflVector:
            self.PrependFunction(builder)(tflItem)

        return builder.EndVector()

class TFLiteAtomicVector(TFLiteVector):
    def __init__(self, vector: list[int, float, bool], StartFunction: Callable[[fb.Builder, int],None]
                , PrependFunction: Callable[[fb.Builder],Callable[[int],None]]) -> None:
        super().__init__(vector,StartFunction,PrependFunction)

    @override
    def genTFLite(self, builder: fb.Builder):
        """ Generates TFLite code for the vector """
        self.StartFunction(builder, len(self.vector))

        # IMPORTANT! tflite MUST be generated for list items in REVERSE ORDER! 
        # Otherwise the order will be wrong.
        for val in reversed(self.vector):
            self.PrependFunction(builder)(val)

        return builder.EndVector()

class FloatVector(TFLiteAtomicVector):
    """ Class represents a TFLite vector of float values. Provides interface for storing data
        and generating output TFLite code. """

    def __init__(self, floatList: list[float], StartFunction: Callable[[fb.Builder, int],None]
    , PrependFunction: Callable[[fb.Builder],None] = lambda builder: builder.PrependFloat32) -> None:
        super().__init__(floatList,StartFunction,PrependFunction)

class IntVector(TFLiteAtomicVector):
    """ Class represents a TFLite vector of integer values. Provides interface for storing data
        and generating output TFLite code. """

    def __init__(self, intList: list[int], StartFunction: Callable[[fb.Builder, int],None]
    , PrependFunction: Callable[[fb.Builder],None] = lambda builder: builder.PrependInt32) -> None:
        super().__init__(intList,StartFunction,PrependFunction)

class BoolVector(TFLiteAtomicVector):
    """ Class represents a TFLite vector of boolean values. Provides interface for storing data
        and generating output TFLite code. """
    def __init__(self, boolList: list[bool], StartFunction: Callable[[fb.Builder, int],None]
    , PrependFunction: Callable[[fb.Builder],None] = lambda builder: builder.PrependBool) -> None:
        super().__init__(boolList,StartFunction,PrependFunction)



class BuiltinOptions(TFLiteObject):
    """ Class represents 'BuiltinOptions' for an Operator. Used in 'model/Operators.py'.
        Provides interface for work with any BuiltinOptions table. 
        This class alone does NOT generate any TFLite.
        Subclasses do NOT generate TFLite for the 'builtinOptionsType', only for the exact options.
        'builtinOptionsType' is merely stored here for convenience and an 'Operator' object
        generates its TFLite representation (as it is the child of the 'operator' table in 'operators'). 
        """
    builtinOptionsType: bo.BuiltinOptions

    def __init__(self, builtinOptionsType: bo.BuiltinOptions) -> None:
        self.builtinOptionsType = builtinOptionsType

    """ Function has to be overwritten """
    def genTFLite(self, builder: fb.Builder):
        err.eprint(f"BuiltinOperator '{self.builtinOptionsType}': genTFLite() is not defined!")
