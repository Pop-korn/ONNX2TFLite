import flatbuffers as fb
from typing import Callable

import tflite.BuiltinOptions as bo

import err

""" This file contains parent classes for simple classes used in the '/model' directory. """

class TFLiteAtomicVector:
    """ Represents a TFLite vector of atomic values. Provides interface for storing data
        and generating output TFLite code. """
    vector: list[int|float|bool] # Item type has to be atomic!

    """ TFLite 'Start...Vector' function for the exact vector. Takes 2 arguments, 
    'floatbuffers.Builder' and number of list elements """
    StartFunction: Callable[[fb.Builder, int],None]

    """ TFLite 'Prepend...' function for the exact vector item type. Takes 'flatbuffers.Builder' 
    as argument """
    PrependFunction: Callable[[fb.Builder],None]

    def __init__(self, list: list, StartFunction: Callable[[fb.Builder, int],None]
                , PrependFunction: Callable[[fb.Builder],None]) -> None:
        self.list = list
        self.StartFunction = StartFunction
        self.PrependFunction = PrependFunction

    def genTFLite(self, builder: fb.Builder):
        """ Generates TFLite code for the vector """
        self.StartFunction(builder, len(self.list))

        for val in self.list:
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



class BuiltinOptions:
    """ Class represents 'BuiltinOptions' for an Operator. Used in 'model/Operators.py'.
        Provides interface for work with any BuiltinOptions table. 
        This class alone does NOT generate any TFLite.
        Subclasses do NOT generate TFLite for the 'builtinOptionsType', only for the exact options,
        'builtinOptionsType' is merely stored here for convenience and an 'Operator' object
        generates its TFLite representation (as it is the child of the 'operator' table in 'operators'). 
        """
    builtinOptionsType: bo.BuiltinOptions

    def __init__(self, builtinOptionsType: bo.BuiltinOptions) -> None:
        self.builtinOptionsType = builtinOptionsType

    """ Function has to be overwritten """
    def genTFLite(self, builder: fb.Builder):
        err.eprint(f"BuiltinOperator '{self.builtinOptionsType}': genTFLite() is not defined")
