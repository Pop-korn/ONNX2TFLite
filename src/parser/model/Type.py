"""
    Type

Representation of ONNX 'Type' objects.
Initialized from a protobuf descriptor.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

import lib.onnx.onnx.onnx_ml_pb2 as onnx

import src.parser.meta.meta as meta

import src.parser.model.TensorShape as ts

class Tensor(meta.ONNXObject):
    elemType: meta.DataType
    shape: ts.TensorShape
    def __init__(self, descriptor: onnx.TypeProto.Tensor) -> None:
        super().__init__(descriptor)
        self.elemType = descriptor.elem_type
        self.shape = ts.TensorShape(descriptor.shape)

class Sequence(meta.ONNXObject):
    elemType: meta.ONNXObject # 'Type' object!
    def __init__(self, descriptor: onnx.TypeProto.Sequence) -> None:
        super().__init__(descriptor)
        self.elemType = Type(descriptor.elem_type)

class Map(meta.ONNXObject):
    keyType: meta.DataType
    valueType: meta.ONNXObject # 'Type' object!
    def __init__(self, descriptor: onnx.TypeProto.Map) -> None:
        super().__init__(descriptor)
        self.keyType = descriptor.key_type
        self.valueType = Type(descriptor.value_type)

class Optional(meta.ONNXObject):
    elemType: meta.ONNXObject # 'Type' object!
    def __init__(self, descriptor: onnx.TypeProto.Optional) -> None:
        super().__init__(descriptor)
        self.elemType = Type(descriptor.elem_type)

class SparseTensor(meta.ONNXObject):
    elemType: meta.DataType
    shape: ts.TensorShape
    def __init__(self, descriptor: onnx.TypeProto.SparseTensor) -> None:
        super().__init__(descriptor)
        self.elemType = descriptor.elem_type
        self.shape = ts.TensorShape(descriptor.shape)

class Opaque(meta.ONNXObject):
    domain: str
    name: str
    def __init__(self, descriptor: onnx.TypeProto.Opaque) -> None:
        super().__init__(descriptor)
        self.domain = descriptor.domain
        self.name = descriptor.name


class Type(meta.ONNXObject):
    denotation: str

    """ The 'Type' object MUST have exactly 1 of these types.
        All unused types are 'None'. """
    tensorType: Tensor
    sequenceType: Sequence
    mapType: Map
    optionalType: Optional
    sparseTensorType: SparseTensor
    opaqueType: Opaque


    def __init__(self, descriptor: onnx.TypeProto) -> None:
        super().__init__(descriptor)
        self.denotation = descriptor.denotation

        # Initialize the types. Only 1 will have a value, others are 'None'.
        self.__resetTypes()
        self.__initUsedType()

    def __initUsedType(self):
        """ Find out which 'type' field was used in the model and initialize the
            corresponding attribute. """
        if self._descriptor.HasField("tensor_type"):
            self.tensorType = Tensor(self._descriptor.tensor_type)
        elif self._descriptor.HasField("sequence_type"):
            self.sequenceType = Sequence(self._descriptor.sequence_type)
        elif self._descriptor.HasField("map_type"):
            self.mapType = Map(self._descriptor.map_type)
        elif self._descriptor.HasField("optional_type"):
            self.optionalType = Optional(self._descriptor.optional_type)
        elif self._descriptor.HasField("sparse_tensor_type"):
            self.sparseTensorType = SparseTensor(self._descriptor.sparse_tensor_type)
        elif self._descriptor.HasField("opaque_type"):
            self.opaqueType = Opaque(self._descriptor.opaque_type)

    def __resetTypes(self):
        """ Set all 'type' attributes to 'None'. """
        self.tensorType = None
        self.sequenceType = None
        self.mapType = None
        self.optionalType = None
        self.sparseTensorType = None
        self.opaqueType = None
