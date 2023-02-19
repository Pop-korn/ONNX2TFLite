import src.generator.model.Tensors as tflT

import src.parser.meta.meta as onnxMeta
import src.parser.model.TensorShape as onnxTS

import lib.tflite.TensorType as tflTT

import src.err as err

def __isNCHW(list: list[int]) -> bool:
    """ Figure out if given 'list' is in the 'nchw' format. """

    # TODO Imporove
    if len(list) >= 4:
        return True

    return False

def __dimsToNHWC(nchwList: list[int]) -> list[int]:
    """ Convert a list of ints which represent dimensions from NCHW to NHWC. """

    res = [nchwList[0]] # First element is 'n'

    channels = nchwList[1] # Save the channels

    res[1:] = nchwList[2:] # Move h,w,... one to the left

    res.append(channels) # Add channels at the end

    return res

def convertShape(oShape: onnxTS.TensorShape) -> tflT.Shape:
    dims = [ dim.value for dim in oShape.dims]

    if __isNCHW(dims):
        dims = __dimsToNHWC(dims)

    return tflT.Shape(dims)

def convertDataType(oType: onnxMeta.DataType) -> tflTT.TensorType:
    match oType:
        case onnxMeta.DataType.UNDEFINED:
            err.wprint("Cannot convert ONNX DataType 'UNDEFINED' to TFLite. Using 'UINT8'.")
            return tflTT.TensorType.UINT8

        case onnxMeta.DataType.FLOAT:
            return tflTT.TensorType.FLOAT32

        case onnxMeta.DataType.UINT8:
            return tflTT.TensorType.UINT8

        case onnxMeta.DataType.INT8:
            return tflTT.TensorType.INT8

        case onnxMeta.DataType.UINT16:
            return tflTT.TensorType.UINT16

        case onnxMeta.DataType.INT16:
            return tflTT.TensorType.INT16

        case onnxMeta.DataType.INT32:
            return tflTT.TensorType.INT32

        case onnxMeta.DataType.INT64:
            return tflTT.TensorType.INT64

        case onnxMeta.DataType.STRING:
            return tflTT.TensorType.STRING

        case onnxMeta.DataType.BOOL:
            return tflTT.TensorType.BOOL

        case onnxMeta.DataType.FLOAT16:
            return tflTT.TensorType.FLOAT16

        case onnxMeta.DataType.DOUBLE:
            return tflTT.TensorType.FLOAT64

        case onnxMeta.DataType.UINT32:
            return tflTT.TensorType.UINT32

        case onnxMeta.DataType.UINT64:
            return tflTT.TensorType.UINT64

        case onnxMeta.DataType.COMPLEX64:
            return tflTT.TensorType.COMPLEX64

        case onnxMeta.DataType.COMPLEX128:
            return tflTT.TensorType.COMPLEX128
            
        case onnxMeta.DataType.BFLOAT16:
            err.wprint("Cannot convert ONNX DataType 'BFLOAT16' to TFLite. Using 'FLOAT16'.")
            return tflTT.TensorType.FLOAT16