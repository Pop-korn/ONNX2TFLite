"""
    Tensors

Representation of ONNX 'Tensor' objects.
Initialized from a protobuf descriptor.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

from typing import List
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
    dims: List[int]
    dataType: meta.DataType
    segment: Segment
    """ Data of the tensor. Shape is given in 'dims' and the ndarray itself MUST BE FLAT.
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

        try:
            self.data =np.reshape(self.data, self.dims)
        except:
            err.error(None,"parser/Tensor.py",
                      f"Could not reshape data of tensor '{self.name}' to shape",
                      f"'{self.dims}'")

    
    def __hasData(self, field):
        """ Determine if given repeated field has data stored in it. """
        return (field is not None) and (field != "") and (len(field) != 0)


    def __assignData(self):
        """ Assign data to either the 'data' attribute correctly. """
        self.data = None

        # Raw data
        if self.__hasData(self._descriptor.raw_data):
            self.data = np.frombuffer(self._descriptor.raw_data, 
                                      types.toNumpyType(self.dataType))
            
            # 'onnx-ml.proto' line '581'
            self.__assertTypeNotBanned([meta.DataType.STRING, 
                                        meta.DataType.UNDEFINED],"raw_data") 
            return
        

        """ 'raw_data' is not given. One of the 'data' fields must contain 
            tensor values. """

        # Float data
        if self.__hasData(self._descriptor.float_data): 
            self.data = np.array(self._descriptor.float_data, 
                                 types.toNumpyType(self.dataType))
            
            # 'onnx-ml.proto' line '540'
            self.__assertTypeAllowed([meta.DataType.FLOAT, 
                                     meta.DataType.COMPLEX64],
                                     "float_data") 

        # Int32 data
        elif self.__hasData(self._descriptor.int32_data): 
            err.unchecked("Tensors.__assignData(): int32_data")
            self.data = np.array(self._descriptor.int32_data, 
                                 types.toNumpyType(self.dataType))
            
            # 'onnx-ml.proto' line '547'
            self.__assertTypeAllowed([meta.DataType.INT32, meta.DataType.INT16, 
                                      meta.DataType.INT8, meta.DataType.UINT16, 
                                      meta.DataType.UINT8, meta.DataType.BOOL, 
                                      meta.DataType.FLOAT16, 
                                      meta.DataType.BFLOAT16],"int32_data") 

        # String data
        elif self.__hasData(self._descriptor.string_data):
            err.unchecked("Tensors.__assignData(): string_data")
            self.data = np.array(self._descriptor.string_data, 
                                 types.toNumpyType(self.dataType))
            
            # 'onnx-ml.proto' line '555'
            self.__assertTypeAllowed([meta.DataType.STRING],"string_data") 

        # Int64 data
        elif self.__hasData(self._descriptor.int64_data):
            self.data = np.array(self._descriptor.int64_data, 
                                 types.toNumpyType(self.dataType))
            
            # 'onnx-ml.proto' line '558'
            self.__assertTypeAllowed([meta.DataType.INT64],"int64_data") 

        # Double data
        elif self.__hasData(self._descriptor.double_data):
            err.unchecked("Tensors.__assignData(): double_data")
            self.data = np.array(self._descriptor.double_data, 
                                 types.toNumpyType(self.dataType))
            
            # 'onnx-ml.proto' line '612'
            self.__assertTypeAllowed([meta.DataType.DOUBLE, 
                                      meta.DataType.COMPLEX128],"double_data") 

        # Uint64 data
        elif self.__hasData(self._descriptor.uint64_data):
            err.unchecked("Tensors.__assignData(): uint64_data")
            self.data = np.array(self._descriptor.uint64_data, 
                                 types.toNumpyType(self.dataType))
            
            # 'onnx-ml.proto' line '617'
            self.__assertTypeAllowed([meta.DataType.UINT32, 
                                      meta.DataType.UINT64],"uint64_data") 


    def __assertTypeAllowed(self, allowedTypes: List, forField: str) -> bool:
        """ Check that 'self.dataType' is in 'allowedTypes'. If it isn't, 
            print warning message.  
            'allowedTypes' is a list of 'meta.DataType' values. 
            Return 'True' if type is allowed. """
        
        if self.dataType not in allowedTypes:
            err.warning(f"ONNX Tensor '{forField}' is used and 'data_type' is",
                        f"'{self.dataType}'! MUST be one of '{allowedTypes}'.")
            err.unchecked("Tensors.__assignData(): uint64_data")
            return False
        
        return True

    def __assertTypeNotBanned(self, bannedTypes: List, forField: str) -> bool:
        """ Check that 'self.dataType' is NOT in 'bannedTypes'. If it IS, print 
            warning message. 
            'bannedTypes' is a list of 'meta.DataType' values. 
            Return 'True' if type is not banned. """
        
        if self.dataType in bannedTypes:
            err.warning(f"ONNX Tensor '{forField}' is used and 'data_type' is",
                        f"'{self.dataType}'! must NOT be one of '{bannedTypes}'.")
            return False
        
        return True



class Tensors(List [Tensor]):
    def __init__(self, descriptor: List[onnx.TensorProto]):
        for item in descriptor:
            self.append(Tensor(item))
