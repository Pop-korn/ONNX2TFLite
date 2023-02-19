# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class HashtableOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = HashtableOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsHashtableOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def HashtableOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # HashtableOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # HashtableOptions
    def TableId(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # HashtableOptions
    def KeyDtype(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # HashtableOptions
    def ValueDtype(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

def Start(builder): builder.StartObject(3)
def HashtableOptionsStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def AddTableId(builder, tableId): builder.PrependInt32Slot(0, tableId, 0)
def HashtableOptionsAddTableId(builder, tableId):
    """This method is deprecated. Please switch to AddTableId."""
    return AddTableId(builder, tableId)
def AddKeyDtype(builder, keyDtype): builder.PrependInt8Slot(1, keyDtype, 0)
def HashtableOptionsAddKeyDtype(builder, keyDtype):
    """This method is deprecated. Please switch to AddKeyDtype."""
    return AddKeyDtype(builder, keyDtype)
def AddValueDtype(builder, valueDtype): builder.PrependInt8Slot(2, valueDtype, 0)
def HashtableOptionsAddValueDtype(builder, valueDtype):
    """This method is deprecated. Please switch to AddValueDtype."""
    return AddValueDtype(builder, valueDtype)
def End(builder): return builder.EndObject()
def HashtableOptionsEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)