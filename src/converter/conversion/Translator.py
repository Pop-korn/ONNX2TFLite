import numpy as np
import functools as ft
from typing import List

import src.generator.model.Tensors as tflT

import src.parser.meta.meta as onnxMeta
import src.parser.model.TensorShape as onnxTS

import src.err as err

import lib.tflite.TensorType as tflTT
import lib.tflite.Padding as tflPad

""" This file contains functions for context-free converting of various
    things from ONNX to TFLite. """


""" -------------------- Private Helper Functions -------------------- """


def isNCHW(dims: List[int]) -> bool:
    """ Figure out if given 'dims' is in the 'nchw' format. """

    # TODO Improve
    if len(dims) >= 4:
        return True

    return False


def __dimsToNCHW(nhwcList: List[int]) -> List[int]:
    """ Convert a list of ints which represent dimensions from NHWC to NCHW. """

    res = [nhwcList[0]] # First element is 'n'

    res.append(nhwcList[-1]) # Channels

    res[2:] = nhwcList[1:-1] # Move h,w,... one to the right

    return res

def __dimsToNHWC(nchwList: List[int]) -> List[int]:
    """ Convert a list of ints which represent dimensions from NCHW to NHWC. """

    res = [nchwList[0]] # First element is 'n'

    channels = nchwList[1] # Save the channels

    res[1:] = nchwList[2:] # Move h,w,... one to the left

    res.append(channels) # Add channels at the end

    return res


def collectionsEqual(colA, colB):
    """ Compare each inidividual element of both collections. 
        They can be any combination of lists, tuples or numpy arrays. 
        Return True if they are equal."""
    if len(colA) != len(colB):
        return False
    
    for (a,b) in zip(colA,colB):
        if a != b:
            return False
    return True


def __isSAMEPadding(oPads: List[int], oKernelShape: List[int],
                    oDilations: List[int]):
    """ Determine if given 'oPads' padding can be represented exactly with the
        'SAME' padding type for given kernel shape. """
    
    """ Calculate the 'range' of the kernel, taking into account dilations and
        kernel shape. """
    if  oDilations is not None and len(oKernelShape) == len(oDilations):
        kernelRange = [ int(dim/2) * dilation for dim, dilation in zip(oKernelShape, oDilations)]
    else:
        kernelRange = [ int(dim/2) for dim in oKernelShape]

    
    for padding, reach in zip(oPads, kernelRange):
        if reach != padding:
            return False
        
    return True




""" -------------------- Public Functions -------------------- """


def shapeFromNumpy(numpyArray):
    """ Return a 'Shape' object representing the shape of given 'numpyArray'. 
    """
    dims = list(numpyArray.shape)
    return tflT.Shape(dims)


def convertPadding(autoPad: str, oPads: List[int], 
                   oKernelShape: List[int],
                   oDilations: List[int]) -> tflPad.Padding:
        """ Convert ONNX pads to TFLite padding. 'autoPad' is the ONNX attribute
            'auto_pad' and 'oPads' is the ONNX attribute 'pads'. 
            The 'oKernelShape' is used to determine if conversion was valid"""

        if autoPad == b"SAME_UPPER":
            return tflPad.Padding.SAME

        elif autoPad == b"SAME_LOWER":
            err.note(f"TFLite does NOT support 'SAME_LOWER' padding!",
                     "Using 'SAME', which is equivalent to 'SAME_UPPER'.")
            return tflPad.Padding.SAME

        elif autoPad == b"VALID":
            return tflPad.Padding.VALID

        # autoPad is NOTSET -> use explicit padding
        if oPads is None:
            err.internal("convertPadding(): oPads is None, when it should not!")
            return tflPad.Padding.VALID

        if all(val == 0 for val in oPads):
            # No padding in any dieraction
            return tflPad.Padding.VALID

        if not __isSAMEPadding(oPads, oKernelShape, oDilations):
            err.warning(f"TFLite does NOT support '{oPads}' padding for kernel",
                        f"'{oKernelShape}' and dilations '{oDilations}'!",
                        "Using 'SAME'.")
        
        return tflPad.Padding.SAME


def convertTensorData(data: np.ndarray, shape: List[int]):
    """ Convert the data of a tensor from the 'NCHW' to 'NHWC' format. """

    err.expectEqualLists(shape, list(data.shape))

    if not isNCHW(shape):
        # 'data' does not need to be converted
        return data

    # Product of all dimensions multiplied together
    size = ft.reduce(lambda a,b : a*b, shape) 
    if size != len(data.flatten()):
        err.error(err.Code.INVALID_TENSOR_SHAPE,
            f"Numpy array for tensor of shape '{shape}' should have '{size}'",
            f"elements, but has '{len(data)}'!")

    # Assign 'data' its current shape
    data.shape = shape

    # "Move" the channels (index 1) to the end
    data = np.moveaxis(data,1,-1)
    
    # Check it worked
    nhwcShape = __dimsToNHWC(shape)
    if not collectionsEqual(data.shape, nhwcShape):
        err.warning(f"Failed to convert data from shape '{shape}'!",
                    f"Got '{data.shape}', expected '{nhwcShape}'.")

    return data


def convertShape(oShape: onnxTS.TensorShape) -> tflT.Shape:
    """ Convert ONNX 'TensorShape', to TFLite 'Shape'. """
    
    dims = [dim.value for dim in oShape.dims]

    return convertShapeDims(dims)


def NHWCShapeToNCHW(nhwcShape: tflT.Shape) -> tflT.Shape:
    """ Create an NCHW version of an NHWC 'generator/shape' object. """

    dims = nhwcShape.vector.copy()
    dims = __dimsToNCHW(dims)

    return tflT.Shape(dims)


def convertShapeDims(oDims: List[int]) -> tflT.Shape:
    """ Convert list of ints representing the shape of an ONNX Tensor
        to a TFLite 'Shape' object. """
    
    dims = [dim for dim in oDims] # Copy just in case

    if isNCHW(dims):
        dims = __dimsToNHWC(dims)

    return tflT.Shape(dims)


def createToNCHWPerm(dims: List[int]) -> np.ndarray:
    """ Take 'dims' in NHWC and return a numpy array, holding data that 
        describes the permutation which would change 'dims' to NCHW. """
    
    perm = __dimsToNCHW( list(range(len(dims))) )

    return np.asarray(perm, np.int32)


def convertDataType(oType: onnxMeta.DataType) -> tflTT.TensorType:
    """ Convert ONNX DataType to TFLite TensorType """

    match oType:
        case onnxMeta.DataType.UNDEFINED:
            err.warning("Cannot convert ONNX DataType 'UNDEFINED' to TFLite.",
                        "Using 'UINT8'.")
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
            err.warning("Cannot convert ONNX DataType 'BFLOAT16' to TFLite.",
                        "Using 'FLOAT16'.")
            return tflTT.TensorType.FLOAT16



def numpyTypeToTFLite(numpyType) -> tflTT.TensorType:
    """ Convert numpy dtype to TFLite TensorType """

    match numpyType:
        case np.float32:
            return tflTT.TensorType.FLOAT32

        case np.uint8:
            return tflTT.TensorType.UINT8

        case np.int8:
            return tflTT.TensorType.INT8

        case np.uint16:
            return tflTT.TensorType.UINT16

        case np.int16:
            return tflTT.TensorType.INT16

        case np.int32:
            return tflTT.TensorType.INT32

        case np.int64:
            return tflTT.TensorType.INT64

        case np.string_:
            return tflTT.TensorType.STRING

        case np.bool_:
            return tflTT.TensorType.BOOL

        case np.float16:
            return tflTT.TensorType.FLOAT16

        case np.float64:
            return tflTT.TensorType.FLOAT64
        case np.double:
            return tflTT.TensorType.FLOAT64

        case np.uint32:
            return tflTT.TensorType.UINT32

        case np.uint64:
            return tflTT.TensorType.UINT64

        case np.complex64:
            return tflTT.TensorType.COMPLEX64

        case np.complex128:
            return tflTT.TensorType.COMPLEX128
            
        case _:
            err.warning(f"Cannot convert numpy data type '{numpyType}'",
                        "to TFLite.")
            return tflTT.TensorType.FLOAT32
