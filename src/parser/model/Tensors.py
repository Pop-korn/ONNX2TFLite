import lib.onnx.onnx.onnx_ml_pb2 as onnx

import src.parser.meta.meta as meta

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
        Type is given in 'dataType'. """
    data: list[ float | int | str ]
    name: str
    docString: str
    """ Alternative to 'data'. Only 1 can be used! """
    rawData: bytes 
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


    def __assignData(self):
        """ Assign data to either the 'data' or 'rawData' attribute correctly. """
        self.rawData = None
        self.data = None
        
        if meta.isDefined(self._descriptor.raw_data):
            # 'raw_data' is present
            self.rawData = self._descriptor.raw_data
            self.__assertTypeNotBanned([meta.DataType.STRING, meta.DataType.UNDEFINED],"raw_data") # 'onnx-ml.proto' line '581'
            return


        # 'raw_data' is not given. One of the 'data' fields must contain tensor values

        if meta.isDefined(self._descriptor.float_data): # TODO Not checked!
            self.data = self._descriptor.float_data
            self.__assertTypeAllowed([meta.DataType.FLOAT, meta.DataType.COMPLEX64],"float_data") # 'onnx-ml.proto' line '540'

        elif meta.isDefined(self._descriptor.int32_data): # TODO Not checked!
            self.data = self._descriptor.int32_data
            self.__assertTypeAllowed([meta.DataType.INT32, meta.DataType.INT16, meta.DataType.INT8, 
                                    meta.DataType.UINT16, meta.DataType.UINT8, meta.DataType.BOOL, 
                                    meta.DataType.FLOAT16, meta.DataType.BFLOAT16],"int32_data") # 'onnx-ml.proto' line '547'

        elif meta.isDefined(self._descriptor.string_data): # TODO Not checked!
            self.data = self._descriptor.string_data
            self.__assertTypeAllowed([meta.DataType.STRING],"string_data") # 'onnx-ml.proto' line '555'

        elif meta.isDefined(self._descriptor.int64_data): # TODO Not checked!
            self.data = self._descriptor.int64_data
            self.__assertTypeAllowed([meta.DataType.INT64],"int64_data") # 'onnx-ml.proto' line '558'

        elif meta.isDefined(self._descriptor.double_data): # TODO Not checked!
            self.data = self._descriptor.double_data
            self.__assertTypeAllowed([meta.DataType.DOUBLE, meta.DataType.COMPLEX128],"double_data") # 'onnx-ml.proto' line '612'

        elif meta.isDefined(self._descriptor.uint64_data): # TODO Not checked!
            self.data = self._descriptor.uint64_data
            self.__assertTypeAllowed([meta.DataType.UINT32, meta.DataType.UINT64],"uint64_data") # 'onnx-ml.proto' line '617'


    def __assertTypeAllowed(self, allowedTypes: list[meta.DataType], forField: str):
        """ Check that 'self.dataType' is in 'allowedTypes'. If it isn't, print warning message. """
        if self.dataType not in allowedTypes:
            err.wprint(f"""ONNX Tensor '{forField}' is used and 'data_type' is '{self.dataType}'!
                MUST be one of '{allowedTypes}'""")

    def __assertTypeNotBanned(self, bannedTypes: list[meta.DataType], forField: str):
        """ Check that 'self.dataType' is NOT in 'bannedTypes'. If it IS, print warning message. """
        if self.dataType in bannedTypes:
            err.wprint(f"""ONNX Tensor '{forField}' is used and 'data_type' is '{self.dataType}'!
                must NOT be one of '{bannedTypes}'""")



class Tensors(list [Tensor]):
    def __init__(self, descriptor: list[onnx.TensorProto]):
        for item in descriptor:
            self.append(Tensor(item))
