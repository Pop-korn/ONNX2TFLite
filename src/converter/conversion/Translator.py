import numpy as np
import functools as ft

import src.generator.model.Tensors as tflT

import src.parser.meta.meta as onnxMeta
import src.parser.model.TensorShape as onnxTS


import src.err as err

import lib.tflite.TensorType as tflTT

""" This file contains functions for context-free converting from ONNX to TFLite. """


""" -------------------- Private Helper Functions -------------------- """


def __isNCHW(list: list[int]) -> bool:
    """ Figure out if given 'list' is in the 'nchw' format. """

    # TODO Improve
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


def __collectionsEqual(colA, colB):
    """ Compare each inidividual element of both collections. 
        They can be any combination of lists, tuples or numpy arrays. 
        Return True if they are equal."""
    for (a,b) in zip(colA,colB):
        if a != b:
            return False
    return True




""" -------------------- Public Functions -------------------- """


def convertTensorData(data: np.ndarray, shape: list[int]):
    """ Convert the data of a tensor from the 'NCHW' to 'NHWC' format. """

    if not __isNCHW(shape):
        # 'data' does not need to be converted
        return data


    size = ft.reduce(lambda a,b : a*b, shape) # Product of all dimensions multiplied together
    if size != len(data):
        err.error(err.Code.INVALID_TENSOR_SHAPE,
            f"Numpy array for tensor of shape '{shape}' should have '{size}' elements, but has '{len(data)}'!",
            "Make sure the 'parser/Tensor.data' is flat. i.e. has no shape!")

    # Assign 'data' its current shape
    data.shape = shape

    # "Move" the channels (index 1) to the end
    data = np.moveaxis(data,1,-1)
    
    # Check it worked
    nhwcShape = __dimsToNHWC(shape)
    if not __collectionsEqual(data.shape, nhwcShape):
        err.warning(f"Failed to convert data from shape '{shape}'! Got '{data.shape}', expected '{nhwcShape}'.")

    return data


def convertShape(oShape: onnxTS.TensorShape) -> tflT.Shape:
    """ Convert ONNX 'TensorShape', to TFLite 'Shape'. """
    dims = [dim.value for dim in oShape.dims]

    return convertShapeDims(dims)


def convertShapeDims(oDims: list[int]) -> tflT.Shape:
    """ Convert list of ints representing the shape of an ONNX Tensor to a TFLite 'Shape' object. """
    dims = [dim for dim in oDims] # Copy just in case

    if __isNCHW(dims):
        dims = __dimsToNHWC(dims)

    return tflT.Shape(dims)


def convertDataType(oType: onnxMeta.DataType) -> tflTT.TensorType:
    """ Convert ONNX DataType to TFLite TensorType """
    match oType:
        case onnxMeta.DataType.UNDEFINED:
            err.warning("Cannot convert ONNX DataType 'UNDEFINED' to TFLite. Using 'UINT8'.")
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
            err.warning("Cannot convert ONNX DataType 'BFLOAT16' to TFLite. Using 'FLOAT16'.")
            return tflTT.TensorType.FLOAT16