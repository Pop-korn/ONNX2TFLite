"""
    types

Module implements functions that work with ONNX data types.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

import numpy as np

import src.parser.meta.meta as meta

import src.err as err


def toNumpyType(oType: meta.DataType):
    """ Convert ONNX DataType to numpy dtype """
    if oType == meta.DataType.UNDEFINED:
        err.warning("Cannot convert ONNX DataType 'UNDEFINED' to numpy dtype. Using 'UINT8'.")
        return np.uint8

    elif oType == meta.DataType.FLOAT:
        return np.float32

    elif oType == meta.DataType.UINT8:
        return np.uint8

    elif oType == meta.DataType.INT8:
        return np.int8

    elif oType == meta.DataType.UINT16:
        return np.uint16

    elif oType == meta.DataType.INT16:
        return np.int16

    elif oType == meta.DataType.INT32:
        return np.int32

    elif oType == meta.DataType.INT64:
        return np.int64

    elif oType == meta.DataType.STRING:
        return np.string_

    elif oType == meta.DataType.BOOL:
        return np.bool_

    elif oType == meta.DataType.FLOAT16:
        return np.float16

    elif oType == meta.DataType.DOUBLE:
        return np.float64

    elif oType == meta.DataType.UINT32:
        return np.uint32

    elif oType == meta.DataType.UINT64:
        return np.uint64

    elif oType == meta.DataType.COMPLEX64:
        return np.cdouble

    elif oType == meta.DataType.COMPLEX128:
        return np.clongdouble

    elif oType == meta.DataType.BFLOAT16:
        err.warning("Cannot convert ONNX DataType 'BFLOAT16' to numpy dtype. Using 'FLOAT16'.")
        return np.uint8
        