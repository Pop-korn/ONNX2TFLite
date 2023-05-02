"""
    types

Module contains helper functions that work with TFLite data types.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""


import flatbuffers as fb

import src.err as err

import lib.tflite.TensorType as tt

def TypeSize(type: tt.TensorType):
    """ Return the memory size in bytes of given TFLite data type. """
    match type:
        case tt.TensorType.UINT8 | tt.TensorType.INT8:
            return 1
        case tt.TensorType.UINT16 | tt.TensorType.INT16 | tt.TensorType.FLOAT16:
            return 2
        case tt.TensorType.UINT32 | tt.TensorType.INT32 | tt.TensorType.FLOAT32:
            return 4
        case tt.TensorType.UINT64 | tt.TensorType.INT64 | tt.TensorType.FLOAT64:
            return 8
        
        # TODO expand
    
    err.warning(f"Unsupported type '{type}'! Assuming 4B size.")
    return 4


def PrependFunction(builder: fb.Builder, type: tt.TensorType):
    """ Return the flatbuffer 'Prepend<type>()' function for given type. """
    match type:
        case tt.TensorType.UINT8:
            return builder.PrependUint8
        case tt.TensorType.UINT16:
            return builder.PrependUint16
        case tt.TensorType.UINT32:
            return builder.PrependUint32
        case tt.TensorType.UINT64:
            return builder.PrependUint64

        case tt.TensorType.INT8:
            return builder.PrependInt8
        case tt.TensorType.INT16:
            return builder.PrependInt16
        case tt.TensorType.INT32:
            return builder.PrependInt32
        case tt.TensorType.INT64:
            return builder.PrependInt64
            
        case tt.TensorType.FLOAT16:
            err.warning("FLOAT16 datatype is not supported! Using default 16b alternative.")
            return builder.PrependInt16 # TODO Might not work
        case tt.TensorType.FLOAT32:
            return builder.PrependFloat32
        case tt.TensorType.FLOAT64:
            return builder.PrependFloat64
        
        # TODO expand
    
    err.warning(f"Unsupported flatbuffer prepend function for type '{type}'!",
                "Using default -> Float32.")
    return builder.PrependFloat32
