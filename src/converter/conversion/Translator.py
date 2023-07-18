"""
    Translator

Module contains functions for context-free conversion of various
things from ONNX to TFLite.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""


import numpy as np
import functools as ft
from typing import List

import src.generator.model.Tensors as tflT

import src.parser.meta.meta as onnxMeta
import src.parser.model.TensorShape as onnxTS

import src.err as err

import lib.tflite.TensorType as tflTT
import lib.tflite.Padding as tflPad


""" -------------------- Private Helper Functions -------------------- """


def isNCHW(dims: List[int]) -> bool:
    """ Figure out if given 'dims' is in the 'nchw' format. """

    # TODO Improve
    # Conv conversion can use NHC / NCH shapes
    if len(dims) >= 4:
        return True

    return False


def isNHWC(dims: List[int]) -> bool:
    """ Figure out if given 'dims' is in the 'nhwc' format. """

    # TODO Improve
    # Conv conversion can use NHC / NCH shapes
    if len(dims) >= 4:
        return True

    return False


def dimsToNCHW(nhwcList: List[int]) -> List[int]:
    """ Convert a list of ints which represent dimensions from NHWC to NCHW. """

    res = list(nhwcList)

    res.insert(1, res.pop()) # Insert 'C' (last item) to index 1

    return res

def dimsToNHWC(nchwList: List[int]) -> List[int]:
    """ Convert a list of ints which represent dimensions from NCHW to NHWC. """

    res = list(nchwList)

    res.append( res.pop(1) ) # Move 'C' (idx 1) to the end

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


def broadcastDims(dims1: List[int], dims2: List[int]):
    """ Broadcast 'dims1' so it matches 'dims2' where possible.
        Same as numpy and onnx broadcasting. """
    
    # TODO expand and improve

    try:

        numDimsToBroadcast = len(dims1)
        newDims = []
        
        for dim in dims2[::-1]:

            if (dims1[numDimsToBroadcast - 1] == dim) and numDimsToBroadcast > 0:
                newDims.insert(0, dim)
                numDimsToBroadcast -= 1

            else:
                newDims.insert(0,1)

        return newDims


    except:
        err.warning(f"Broadcasting of shapes '{dims1}' to  '{dims2}'",
                    "is not yet implemented!")
        return dims1



def permutationsAreInverse(perm1: List[int], perm2: List[int]) -> bool:
    """ Determine if given Transpose permutations are inverse of each other. 
        i.e. when applied back to back, there will be no effect. """
    # Example:
    #   0 3 1 2
    #   0 2 3 1

    for i, _ in enumerate(perm1):
        if i != perm1[perm2[i]]:
            return False
        
    return True


def nchToNhwcDims(nchDims: List[int]):
    """ Convert a list of ints representing the shape of an NCH tensor to the
        dimensions of an equivalent NHWC tensor. """

    res = nchDims.copy()

    res.append( res.pop(1) ) # Move 'C' to the end

    res.insert(2,1) # Insert 'W' = 1
    
    return res


def nchToNchwDims(nchDims: List[int]):
    """ Convert a list of ints representing the shape of an NCH tensor to the
        dimensions of an equivalent NCHW tensor. i.e. add '1' to the end. """

    res = nchDims.copy()

    res.append(1)
    
    return res


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
                        "Using padding 'SAME', which may cause issues.")
        
        return tflPad.Padding.SAME


def nchToNhwcData(data: np.ndarray, dims: List[int]):
    """ Convert the data of an 'NCH' tensor to equivalent 'NHWC' format. """

    dims = dims.copy()

    err.expectEqualLists(dims, list(data.shape))

    # Product of all dimensions multiplied together (total size)
    size = ft.reduce(lambda a,b : a*b, dims) 
    if size != len(data.flatten()):
        err.error(err.Code.INVALID_TENSOR_SHAPE,
            f"Numpy array for tensor of shape '{dims}' should have '{size}'",
            f"elements, but has '{len(data)}'!")
        
    # Append '1' to the end of the shape to make it NCHW
    dims.append(1)

    # Assign 'data' its current shape 
    data.shape = dims

    # "Move" the channels (index 1) to the end to make it NHWC
    data = np.moveaxis(data,1,-1)

    # Check it worked
    nhwcShape = dimsToNHWC(dims)
    if not collectionsEqual(data.shape, nhwcShape):
        err.warning(f"Failed to convert data from NCH shape '{dims}' to NHC!",
                    f"Got '{data.shape}', expected '{nhwcShape}'.")

    return data

def convertTensorData(data: np.ndarray, shape: List[int]):
    """ Convert the data of a tensor from the 'NCHW' to 'NHWC' format. """

    err.expectEqualLists(shape, list(data.shape))

    if not isNCHW(shape):
        # 'data' does not need to be converted
        return data

    # Product of all dimensions multiplied together (total size)
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
    nhwcShape = dimsToNHWC(shape)
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
    dims = dimsToNCHW(dims)

    return tflT.Shape(dims)


def convertShapeDims(oDims: List[int]) -> tflT.Shape:
    """ Convert list of ints representing the shape of an ONNX Tensor
        to a TFLite 'Shape' object. """
    
    dims = [dim for dim in oDims] # Copy just in case

    if isNCHW(dims):
        dims = dimsToNHWC(dims)

    return tflT.Shape(dims)


def createToNCHWPerm(dims: List[int]) -> np.ndarray:
    """ Take 'dims' in NHWC and return a numpy array, holding data that 
        describes the permutation which would change 'dims' to NCHW. """
    
    perm = dimsToNCHW( list(range(len(dims))) )

    return np.asarray(perm, np.int32)

def createToNHWCPerm(dims: List[int]) -> np.ndarray:
    """ Take 'dims' in NCHW and return a numpy array, holding data that 
        describes the permutation which would change 'dims' to NHWC. """
    
    perm = dimsToNHWC( list(range(len(dims))) )

    return np.asarray(perm, np.int32)


def createAxisToLastPerm(axis, numDims):
    """ Create a numpy array representing the transpose permutations needed, to 
        make the 'axis' dimension, the last dimension. """
    
    dims = list(range(numDims))
    
    if axis == numDims-1:
        return dims
    elif axis >= numDims or axis < 0:
        err.warning(f"Translator.createAxisToLastPerm({axis},{numDims}).",
                    "Inputs don't make sense!")
        return np.asarray(dims, np.int32)

    # Remember axis dimension
    axisDim = dims[axis]

    # Move dimensions after 'axis' to the left
    dims[axis:-1] = dims[axis+1:-1]

    # Add axis dimension to the end
    dims.append(axisDim)
    
    return np.asarray(dims, np.int32)


def convertDataType(oType: onnxMeta.DataType) -> tflTT.TensorType:
    """ Convert ONNX DataType to TFLite TensorType """

    if oType == onnxMeta.DataType.UNDEFINED:
        err.warning("Cannot convert ONNX DataType 'UNDEFINED' to TFLite.",
                    "Using 'UINT8'.")
        return tflTT.TensorType.UINT8

    elif oType == onnxMeta.DataType.FLOAT:
        return tflTT.TensorType.FLOAT32

    elif oType == onnxMeta.DataType.UINT8:
        return tflTT.TensorType.UINT8

    elif oType == onnxMeta.DataType.INT8:
        return tflTT.TensorType.INT8

    elif oType == onnxMeta.DataType.UINT16:
        return tflTT.TensorType.UINT16

    elif oType == onnxMeta.DataType.INT16:
        return tflTT.TensorType.INT16

    elif oType == onnxMeta.DataType.INT32:
        return tflTT.TensorType.INT32

    elif oType == onnxMeta.DataType.INT64:
        return tflTT.TensorType.INT64

    elif oType == onnxMeta.DataType.STRING:
        return tflTT.TensorType.STRING

    elif oType == onnxMeta.DataType.BOOL:
        return tflTT.TensorType.BOOL

    elif oType == onnxMeta.DataType.FLOAT16:
        return tflTT.TensorType.FLOAT16

    elif oType == onnxMeta.DataType.DOUBLE:
        return tflTT.TensorType.FLOAT64

    elif oType == onnxMeta.DataType.UINT32:
        return tflTT.TensorType.UINT32

    elif oType == onnxMeta.DataType.UINT64:
        return tflTT.TensorType.UINT64

    elif oType == onnxMeta.DataType.COMPLEX64:
        return tflTT.TensorType.COMPLEX64

    elif oType == onnxMeta.DataType.COMPLEX128:
        return tflTT.TensorType.COMPLEX128

    elif oType == onnxMeta.DataType.BFLOAT16:
        err.warning("Cannot convert ONNX DataType 'BFLOAT16' to TFLite.",
                    "Using 'FLOAT16'.")
        return tflTT.TensorType.FLOAT16



def numpyTypeToTFLite(numpyType) -> tflTT.TensorType:
    """ Convert numpy dtype to TFLite TensorType """

    if numpyType == np.float32:
        return tflTT.TensorType.FLOAT32

    elif numpyType == np.uint8:
        return tflTT.TensorType.UINT8

    elif numpyType == np.int8:
        return tflTT.TensorType.INT8

    elif numpyType == np.uint16:
        return tflTT.TensorType.UINT16

    elif numpyType == np.int16:
        return tflTT.TensorType.INT16

    elif numpyType == np.int32:
        return tflTT.TensorType.INT32

    elif numpyType == np.int64:
        return tflTT.TensorType.INT64

    elif numpyType == np.string_:
        return tflTT.TensorType.STRING

    elif numpyType == np.bool_:
        return tflTT.TensorType.BOOL

    elif numpyType == np.float16:
        return tflTT.TensorType.FLOAT16

    elif numpyType == np.float64:
        return tflTT.TensorType.FLOAT64
    elif numpyType == np.double:
        return tflTT.TensorType.FLOAT64

    elif numpyType == np.uint32:
        return tflTT.TensorType.UINT32

    elif numpyType == np.uint64:
        return tflTT.TensorType.UINT64

    elif numpyType == np.complex64:
        return tflTT.TensorType.COMPLEX64

    elif numpyType == np.complex128:
        return tflTT.TensorType.COMPLEX128

    else:
        err.warning(f"Cannot convert numpy data type '{numpyType}'",
                    "to TFLite.")
        return tflTT.TensorType.FLOAT32
        

def TFLiteTypeToNumpy(tflType: tflTT.TensorType) -> np.dtype:
    """ Convert TFLite TensorType to numpy dtype """

    if tflType == tflTT.TensorType.FLOAT32:
        return np.float32

    elif tflType == tflTT.TensorType.UINT8:
        return np.uint8

    elif tflType == tflTT.TensorType.INT8:
        return np.int8

    elif tflType == tflTT.TensorType.UINT16:
        return np.uint16

    elif tflType == tflTT.TensorType.INT16:
        return np.int16

    elif tflType == tflTT.TensorType.INT32:
        return np.int32

    elif tflType == tflTT.TensorType.INT64:
        return np.int64

    elif tflType == tflTT.TensorType.STRING:
        return np.string_

    elif tflType == tflTT.TensorType.BOOL:
        return np.bool_

    elif tflType == tflTT.TensorType.FLOAT16:
        return np.float16

    elif tflType == tflTT.TensorType.FLOAT64:
        return np.float64

    elif tflType == tflTT.TensorType.UINT32:
        return np.uint32

    elif tflType == tflTT.TensorType.UINT64:
        return np.uint64

    elif tflType == tflTT.TensorType.COMPLEX64:
        return np.complex64

    elif tflType == tflTT.TensorType.COMPLEX128:
        return np.complex128

    else:
        err.warning(f"Cannot convert TFLite type '{tflType}'",
                    "to numpy dtype.")
        return np.float32
