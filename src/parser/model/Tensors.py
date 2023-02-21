import numpy as np

import lib.onnx.onnx.onnx_ml_pb2 as onnx

import src.parser.meta.meta as meta
import src.parser.meta.types as types

import src.err as err

class Segment:
    begin: int
    end: int

    def __init__(self, descriptor: onnx.TensorProto.Segment) -> None:
        self.begin = descriptor.begin
        self.end = descriptor.end

class Tensor(meta.ONNXObject):
    dims: list[int]
    dataType: meta.DataType
    segment: Segment
    """ Data of the tensor. Can be of type float, double, int32, int64, uint64 or string.
        Type is given in 'dataType' and should also be specified in the 'data.dtype'. """
    data: np.ndarray
    name: str
    docString: str
    # TODO externalData
    dataLocation: meta.DataLocation

    def __init__(self, descriptor: onnx.TensorProto) -> None:
        super().__init__(descriptor)
        self.dims = descriptor.dims
        self.dataType = descriptor.data_type
        self.segment = Segment(descriptor.segment)
        self.name = descriptor.name
        self.docString = descriptor.doc_string
        self.dataLocation = descriptor.data_location
        self.__assignData()

    
    def __hasRawData(self):
        """ Figure out if 'raw_data' field is present in the '_descriptor'. """
        return meta.isDefined(self._descriptor.raw_data) and len(self._descriptor.raw_data) > 0


    def __assignData(self):
        """ Assign data to either the 'data' or 'rawData' attribute correctly. """
        self.rawData = None
        self.data = None

        # Raw data
        if self.__hasRawData():
            self.data = np.frombuffer(self._descriptor.raw_data, types.toNumpyType(self.dataType))
            self.__assertTypeNotBanned([meta.DataType.STRING, meta.DataType.UNDEFINED],"raw_data") # 'onnx-ml.proto' line '581'
            return
        

        """ 'raw_data' is not given. One of the 'data' fields must contain tensor values. """

        # Float data
        if meta.isDefined(self._descriptor.float_data): 
            self.data = np.array(self._descriptor.float_data, types.toNumpyType(self.dataType))
            self.__assertTypeAllowed([meta.DataType.FLOAT, meta.DataType.COMPLEX64],"float_data") # 'onnx-ml.proto' line '540'

        # Int32 data
        elif meta.isDefined(self._descriptor.int32_data): 
            err.unchecked("Tensors.__assignData(): int32_data")
            self.data = np.array(self._descriptor.int32_data, types.toNumpyType(self.dataType))
            self.__assertTypeAllowed([meta.DataType.INT32, meta.DataType.INT16, meta.DataType.INT8, 
                                    meta.DataType.UINT16, meta.DataType.UINT8, meta.DataType.BOOL, 
                                    meta.DataType.FLOAT16, meta.DataType.BFLOAT16],"int32_data") # 'onnx-ml.proto' line '547'

        # String data
        elif meta.isDefined(self._descriptor.string_data):
            err.unchecked("Tensors.__assignData(): string_data")
            self.data = np.array(self._descriptor.string_data, types.toNumpyType(self.dataType))
            self.__assertTypeAllowed([meta.DataType.STRING],"string_data") # 'onnx-ml.proto' line '555'

        # Int64 data
        elif meta.isDefined(self._descriptor.int64_data):
            err.unchecked("Tensors.__assignData(): int64_data")
            self.data = np.array(self._descriptor.int64_data, types.toNumpyType(self.dataType))
            self.__assertTypeAllowed([meta.DataType.INT64],"int64_data") # 'onnx-ml.proto' line '558'

        # Double data
        elif meta.isDefined(self._descriptor.double_data):
            err.unchecked("Tensors.__assignData(): double_data")
            self.data = np.array(self._descriptor.double_data, types.toNumpyType(self.dataType))
            self.__assertTypeAllowed([meta.DataType.DOUBLE, meta.DataType.COMPLEX128],"double_data") # 'onnx-ml.proto' line '612'

        # Uint64 data
        elif meta.isDefined(self._descriptor.uint64_data):
            err.unchecked("Tensors.__assignData(): uint64_data")
            self.data = np.array(self._descriptor.uint64_data, types.toNumpyType(self.dataType))
            self.__assertTypeAllowed([meta.DataType.UINT32, meta.DataType.UINT64],"uint64_data") # 'onnx-ml.proto' line '617'


    def __assertTypeAllowed(self, allowedTypes: list[meta.DataType], forField: str):
        """ Check that 'self.dataType' is in 'allowedTypes'. If it isn't, print warning message. """
        if self.dataType not in allowedTypes:
            err.warning(f"""ONNX Tensor '{forField}' is used and 'data_type' is '{self.dataType}'!
                MUST be one of '{allowedTypes}'""")

    def __assertTypeNotBanned(self, bannedTypes: list[meta.DataType], forField: str):
        """ Check that 'self.dataType' is NOT in 'bannedTypes'. If it IS, print warning message. """
        if self.dataType in bannedTypes:
            err.warning(f"""ONNX Tensor '{forField}' is used and 'data_type' is '{self.dataType}'!
                must NOT be one of '{bannedTypes}'""")



class Tensors(list [Tensor]):
    def __init__(self, descriptor: list[onnx.TensorProto]):
        for item in descriptor:
            self.append(Tensor(item))
