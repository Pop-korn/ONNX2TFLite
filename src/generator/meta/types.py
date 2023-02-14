import flatbuffers as fb

import err

import tflite.TensorType as tt

def TypeSize(type: tt.TensorType):
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
    
    err.wprint(f"Unsupported type '{type}'! Assuming 4B size.")
    return 4


def PrependFunction(builder: fb.Builder, type: tt.TensorType):
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
            err.wprint("FLOAT16 datatype is not supported! Using default 16b alternative.")
            return builder.PrependInt16 # TODO Might not work
        case tt.TensorType.FLOAT32:
            return builder.PrependFloat32
        case tt.TensorType.FLOAT64:
            return builder.PrependFloat64
        
        # TODO expand
    
    err.wprint(f"Unsupported type '{type}'! Using default Float32.")
    return builder.PrependFloat32
