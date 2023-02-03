import flatbuffers as fb
from typing import Callable

""" This file contains parent classes for simple classes used in the '/model' directory. """



class FloatVector:
    """ Class represents a TFLite vector of float values. Provides interface for storing data
        and generating output TFLite code. """

    floatList: list[float]

    """ TFLite 'Start...Vector' function for the exact vector. Takes 2 arguments, 
    'floatbuffers.Builder' and number of list elements """
    StartFunction: Callable[[fb.Builder, int],None]

    """ TFLite 'Prepend...' function for the exact vector item type. Takes 'flatbuffers.Builder' 
    as argument """
    PrependFunction: Callable[[fb.Builder],None]

    def __init__(self, floatList: list[float], StartFunction: Callable[[fb.Builder, int],None]
    , PrependFunction: Callable[[fb.Builder],None] = lambda builder: builder.PrependFloat32) -> None:
        self.floatList = floatList
        self.StartFunction = StartFunction
        self.PrependFunction = PrependFunction

    def genTFLite(self, builder: fb.Builder):
        """ Generates TFLite code for the vector """
        self.StartFunction(builder, len(self.floatList))

        for floatVal in self.floatList:
            self.PrependFunction(builder)(floatVal)

        return builder.EndVector()



class IntVector:
    """ Class represents a TFLite vector of integer values. Provides interface for storing data
        and generating output TFLite code. """

    intList: list[int]

    """ TFLite 'Start...Vector' function for the exact vector. Takes 2 arguments,
    'floatbuffers.Builder' and number of list elements """
    StartFunction: Callable[[fb.Builder, int],None] # TFLite 'Start' function for the exact vector
    
    """ TFLite 'Prepend...' function for the exact vector item type. Takes 'flatbuffers.Builder' 
    as argument """
    PrependFunction: Callable[[fb.Builder],None] # TFLite 'Prepend' function for the exact vector item

    def __init__(self, intList: list[int], StartFunction: Callable[[fb.Builder, int],None]
    , PrependFunction: Callable[[fb.Builder],None] = lambda builder: builder.PrependInt32) -> None:
        self.intList = intList
        self.StartFunction = StartFunction
        self.PrependFunction = PrependFunction

    def genTFLite(self, builder: fb.Builder):
        """ Generates TFLite code for the vector """
        self.StartFunction(builder, len(self.intList))

        for intVal in self.intList:
            self.PrependFunction(builder)(intVal)

        return builder.EndVector()
