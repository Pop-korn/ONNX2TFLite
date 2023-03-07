from typing_extensions import override
import flatbuffers as fb
from typing import Callable, List

import lib.tflite.BuiltinOptions as bOpt
import lib.tflite.BuiltinOperator as bOp

import src.err as err

""" This file contains parent classes for simple classes used in the '/model' directory. """

class TFLiteObject:
    """ Parent class for all tflite objects. That is all objects in the 'generator' directory. """
    
    """ Generates tflite representation for this object. MUST be overriden! """
    def genTFLite(self, builder: fb.Builder):
        err.warning("TFLiteObject: genTFLite() is not defined!")

class TFLiteVector(TFLiteObject):
    """ Represents a TFLite vector of TFLiteObjects. Provides interface for storing data
        and generating output TFLite code. """

    vector: List[TFLiteObject]

    """ Indicates if an empty vector should be generated if 'vector' attribute is
    empty, or to not generate anything in that case. """
    genEmpty: bool=True

    """ TFLite 'Start...Vector' function for the exact vector. Takes 2 arguments, 
    'floatbuffers.Builder' and number of vector elements """
    StartFunction: Callable[[fb.Builder, int],None]

    """ TFLite 'Prepend...' function for the exact vector item type. Takes 'flatbuffers.Builder' 
    as argument """
    PrependFunction: Callable[[fb.Builder],None]

    def __init__(self, vector: List[TFLiteObject], 
                StartFunction: Callable[[fb.Builder, int],None],
                PrependFunction: Callable[[fb.Builder],Callable[[int],None]] = lambda builder: builder.PrependUOffsetTRelative,
                genEmpty: bool=True) -> None:
        if vector is None:
            vector = []
        self.vector = vector
        self.StartFunction = StartFunction
        self.PrependFunction = PrependFunction
        self.genEmpty = genEmpty

    def append(self, item):
        self.vector.append(item)

    def remove(self, item):
        self.vector.remove(item)

    def get(self, index: int):
        return self.vector[index]
    
    def getLast(self):
        if len(self.vector) > 0:
            return self.vector[-1]
        return None

    def len(self):
        return self.vector.__len__()

    def genTFLite(self, builder: fb.Builder):
        """ Generates TFLite code for the vector """

        if (not self.genEmpty) and (len(self.vector) == 0):
            # Nothing to generate
            return

        # IMPORTANT! tflite MUST be generated for list items in REVERSE ORDER! 
        # Otherwise the order will be wrong.
        tflVector = [item.genTFLite(builder) for item in reversed(self.vector)]

        self.StartFunction(builder, len(self.vector))

        for tflItem in tflVector:
            self.PrependFunction(builder)(tflItem)

        return builder.EndVector()

class TFLiteAtomicVector(TFLiteVector):
    def __init__(self, vector: List[int | float | bool],
                StartFunction: Callable[[fb.Builder, int],None],
                PrependFunction: Callable[[fb.Builder],Callable[[int],None]],
                genEmpty: bool=True) -> None:
        super().__init__(vector,StartFunction,PrependFunction,genEmpty)

    @override
    def genTFLite(self, builder: fb.Builder):
        """ Generates TFLite code for the vector """

        if (not self.genEmpty) and (len(self.vector) == 0):
            # Nothing to generate
            return

        self.StartFunction(builder, len(self.vector))

        # IMPORTANT! tflite MUST be generated for list items in REVERSE ORDER! 
        # Otherwise the order will be wrong.
        for val in reversed(self.vector):
            self.PrependFunction(builder)(val)

        return builder.EndVector()

class FloatVector(TFLiteAtomicVector):
    """ Class represents a TFLite vector of float values. Provides interface for storing data
        and generating output TFLite code. """

    def __init__(self, floatList: List[float], 
                StartFunction: Callable[[fb.Builder, int],None],
                PrependFunction: Callable[[fb.Builder],None] = lambda builder: builder.PrependFloat32,
                genEmpty: bool=True) -> None:
        super().__init__(floatList,StartFunction,PrependFunction,genEmpty)

class IntVector(TFLiteAtomicVector):
    """ Class represents a TFLite vector of integer values. Provides interface for storing data
        and generating output TFLite code. """

    def __init__(self, intList: List[int], 
                StartFunction: Callable[[fb.Builder, int],None],
                PrependFunction: Callable[[fb.Builder],None] = lambda builder: builder.PrependInt32,
                genEmpty: bool=True) -> None:
        super().__init__(intList,StartFunction,PrependFunction,genEmpty)

class BoolVector(TFLiteAtomicVector):
    """ Class represents a TFLite vector of boolean values. Provides interface for storing data
        and generating output TFLite code. """
    def __init__(self, boolList: List[bool],
                StartFunction: Callable[[fb.Builder, int],None],
                PrependFunction: Callable[[fb.Builder],None] = lambda builder: builder.PrependBool,
                genEmpty: bool=True) -> None:
        super().__init__(boolList,StartFunction,PrependFunction,genEmpty)



class BuiltinOptions(TFLiteObject):
    """ Class represents 'BuiltinOptions' for an Operator. Used in 'model/Operators.py'.
        Provides interface for work with any BuiltinOptions table. 
        This class alone does NOT generate any TFLite.
        Subclasses do NOT generate TFLite for the 'builtinOptionsType', only for the exact options.
        'builtinOptionsType' is merely stored here for convenience and an 'Operator' object
        generates its TFLite representation (as it is the child of the 'operator' table in 'operators'). 
        """

    """ The type of parameters of this operator. """
    builtinOptionsType: bOpt.BuiltinOptions

    """ The type of this operator. """
    operatorType: bOp.BuiltinOperator

    def __init__(self, builtinOptionsType: bOpt.BuiltinOptions, 
                 operatorType: bOp.BuiltinOperator) -> None:
        if builtinOptionsType is None:
            err.internal("TFLITE: Operator inheritting from 'BuiltinOptions'",
                         "MUST specify the 'builtinOptionsType'!")
        if operatorType is None:
            err.internal("TFLITE: Operator inheritting from 'BuiltinOptions'",
                         "MUST specify the 'operatorType'!")
        self.builtinOptionsType = builtinOptionsType
        self.operatorType = operatorType

    """ Function has to be overwritten """
    def genTFLite(self, builder: fb.Builder):
        err.warning(f"BuiltinOperator '{self.builtinOptionsType}':",
                    "genTFLite() is not defined!")
