import onnx.onnx.onnx_ml_pb2 as onnx

import parser.meta.meta as meta

import err

class Dimention(meta.ONNXObject):
    value: int | str
    denotation: str

    def __init__(self, descriptor: onnx.TensorShapeProto.Dimension) -> None:
        super().__init__(descriptor)
        self.denotation = descriptor.denotation

        if descriptor.HasField("dim_value"):
            self.value = descriptor.dim_value
        elif descriptor.HasField("dim_param"):
            self.value = descriptor.dim_param
        else:
            err.wprint("ONNX TensorShape.Dimension has no valid value!")

class TensorShape(meta.ONNXObject):
    dims: list[Dimention]

    def __init__(self, descriptor: onnx.TensorShapeProto) -> None:
        super().__init__(descriptor)
        self.dims = []
        for item in descriptor.dim:
            self.dims.append(Dimention(item))




class Type(meta.ONNXObject):
    denotation: str

    """ The 'Type' object MUST have exactly 1 of these types.
        All unused types are 'None'. """
    tensorType: meta.ONNXObject # Tensor
    sequenceType: meta.ONNXObject # Sequence
    mapType: meta.ONNXObject # Map
    optionalType: meta.ONNXObject # Optional
    sparseTensorType: meta.ONNXObject # SparseTensor
    opaqueType: meta.ONNXObject # Opaque


    def __init__(self, descriptor: onnx.TypeProto) -> None:
        super().__init__(descriptor)
        self.denotation = descriptor.denotation

        # Initialize the types. Only 1 will have a value, others are 'None'.
        self.__resetTypes()
        self.__initUsedType()

    def __initUsedType(self):
        if self._descriptor.HasField("tensor_type"):
            self.tensorType = self.Tensor(self._descriptor.tensor_type)
        elif self._descriptor.HasField("sequence_type"):
            self.sequenceType = self.Sequence(self._descriptor.sequence_type)
        elif self._descriptor.HasField("map_type"):
            self.mapType = self.Map(self._descriptor.map_type)
        elif self._descriptor.HasField("optional_type"):
            self.optionalType = self.Optional(self._descriptor.optional_type)
        elif self._descriptor.HasField("sparse_tensor_type"):
            self.sparseTensorType = self.SparseTensor(self._descriptor.sparse_tensor_type)
        elif self._descriptor.HasField("opaque_type"):
            self.opaqueType = self.Opaque(self._descriptor.opaque_type)

    def __resetTypes(self):
        self.tensorType = None
        self.sequenceType = None
        self.mapType = None
        self.optionalType = None
        self.sparseTensorType = None
        self.opaqueType = None


    """ Classes cross-referrencing with 'Type'. """

    class Tensor(meta.ONNXObject):
        elemType: meta.DataType
        shape: TensorShape

        def __init__(self, descriptor: onnx.TypeProto.Tensor) -> None:
            super().__init__(descriptor)
            self.elemType = descriptor.elem_type
            self.shape = TensorShape(descriptor.shape)

    class Sequence(meta.ONNXObject):
        elemType: meta.ONNXObject # Type object

        def __init__(self, descriptor: onnx.TypeProto.Sequence) -> None:
            super().__init__(descriptor)
            self.elemType = Type(descriptor.elem_type)

    class Map(meta.ONNXObject):
        keyType: meta.DataType
        valueType: meta.ONNXObject # Type object

        def __init__(self, descriptor: onnx.TypeProto.Map) -> None:
            super().__init__(descriptor)
            self.keyType = descriptor.key_type
            self.valueType = Type(descriptor.value_type)

    class Optional(meta.ONNXObject):
        elemType: meta.ONNXObject # Type object

        def __init__(self, descriptor: onnx.TypeProto.Optional) -> None:
            super().__init__(descriptor)
            self.elemType = Type(descriptor.elem_type)

    class SparseTensor(meta.ONNXObject):
        elemType: meta.DataType
        shape: TensorShape

        def __init__(self, descriptor: onnx.TypeProto.SparseTensor) -> None:
            super().__init__(descriptor)
            self.elemType = descriptor.elem_type
            self.shape = TensorShape(descriptor.shape)

    class Opaque(meta.ONNXObject):
        domain: str
        name: str

        def __init__(self, descriptor: onnx.TypeProto.Opaque) -> None:
            super().__init__(descriptor)
            self.domain = descriptor.domain
            self.name = descriptor.name
