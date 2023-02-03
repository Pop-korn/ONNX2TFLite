import flatbuffers as fb
""" This file contains parent classes for simple classes used in the /model directory. """

class FloatVector:
    floatList: list[float]
    StartFunction: callable
    PrependFunction: callable(fb.Builder)

    def __init__(self, floatList: list[float], StartFunction: callable
    , PrependFunction: callable(fb.Builder) = lambda builder: builder.PrependFloat32) -> None:
        self.floatList = floatList
        self.StartFunction = StartFunction
        self.PrependFunction = PrependFunction

    def genTFLite(self, builder: fb.Builder):
        self.StartFunction(builder, len(self.floatList))

        for floatVal in self.floatList:
            self.PrependFunction(builder)(floatVal)

        return builder.EndVector()

class IntVector:
    intList: list[int]
    StartFunction: callable
    PrependFunction: callable(fb.Builder)

    def __init__(self, intList: list[int], StartFunction: callable
    , PrependFunction: callable(fb.Builder) = lambda builder: builder.PrependInt32) -> None:
        self.intList = intList
        self.StartFunction = StartFunction
        self.PrependFunction = PrependFunction

    def genTFLite(self, builder: fb.Builder):
        self.StartFunction(builder, len(self.intList))

        for intVal in self.intList:
            self.PrependFunction(builder)(intVal)

        return builder.EndVector()
