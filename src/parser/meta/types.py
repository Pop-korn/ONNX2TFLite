import numpy as np

import src.parser.meta.meta as meta

import src.err as err

def toNumpyType(oType: meta.DataType):
    """ Convert ONNX DataType to numpy dtype """
    match oType:
        case meta.DataType.UNDEFINED:
            err.warning("Cannot convert ONNX DataType 'UNDEFINED' to numpy dtype. Using 'UINT8'.")
            return np.uint8

        case meta.DataType.FLOAT:
            return np.float32

        case meta.DataType.UINT8:
            return np.uint8

        case meta.DataType.INT8:
            return np.int8

        case meta.DataType.UINT16:
            return np.uint16

        case meta.DataType.INT16:
            return np.int16

        case meta.DataType.INT32:
            return np.int32

        case meta.DataType.INT64:
            return np.int64

        case meta.DataType.STRING:
            return np.string_

        case meta.DataType.BOOL:
            return np.bool_

        case meta.DataType.FLOAT16:
            return np.float16

        case meta.DataType.DOUBLE:
            return np.float64

        case meta.DataType.UINT32:
            return np.uint32

        case meta.DataType.UINT64:
            return np.uint64

        case meta.DataType.COMPLEX64:
            return np.cdouble

        case meta.DataType.COMPLEX128:
            return np.clongdouble
            
        case meta.DataType.BFLOAT16:
            err.warning("Cannot convert ONNX DataType 'BFLOAT16' to numpy dtype. Using 'FLOAT16'.")
            return np.uint8