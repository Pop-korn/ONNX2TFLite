from typing import Callable

import src.generator.meta.meta as tflMeta
import src.generator.model.Operators as tflO
import src.generator.model.Tensors as tflT
import src.generator.model.Buffers as tflB
import src.generator.builtin.Conv2D as tflConv2D
import src.generator.builtin.LRN as tflLRN
import src.generator.builtin.MaxPool2D as tflMaxPool2D
import src.generator.builtin.Reshape as tflReshape

import src.parser.model.Nodes as onnxN
import src.parser.builtin.Conv as onnxConv
import src.parser.builtin.LRN as onnxLRN
import src.parser.builtin.MaxPool as onnxMaxPool
import src.parser.builtin.Reshape as onnxReshape

import src.err as err

import lib.tflite.Padding as tflPad
import lib.tflite.BuiltinOperator as tflBO


""" -------------------- Helper Operator Functions -------------------- """

def __isOfSize(obj, size: int):
    if obj is None:
        return False

    return len(obj) == size


def __isSAMEPadding(oPads: list[int], oKernelShape: list[int]):
    """ Determine if given 'oPads' padding can be represented exactly with the
        'SAME' padding type for given kernel shape. """

    for padding, dim in zip(oPads, oKernelShape):
        if dim // 2 != padding:
            return False

    return True


def __assign2DStrides(obj, strides: list[int]):
    """ Assign the 'obj' attributes 'strideH' and 'strideW' from 'strides'.
        'obj' MUST have these attributes. """
    if __isOfSize(strides, 2):
        obj.strideH = strides[0]
        obj.strideW = strides[1]
    else:
        err.note(f"Expected 2D strides, got '{strides}'. Leaving default values.")
    


def __convertPadding(autoPad: str, oPads: list[int], oKernelShape: list[int]) -> tflPad.Padding:
    """ Convert ONNX pads to TFLite padding. 
        'autoPad' is the ONNX attribute 'auto_pad' and 'oPads' is the ONNX attribute 'pads'. 
        The 'oKernelShape' is used to determine if conversion was valid"""

    if autoPad == "SAME_UPPER":
        return tflPad.Padding.SAME

    elif autoPad == "SAME_LOWER":
        err.note(f"TFLite does NOT support 'SAME_LOWER' padding! Using 'SAME', which is equivalent to 'SAME_UPPER'.")
        return tflPad.Padding.SAME

    elif autoPad == "VALID":
        return tflPad.Padding.VALID

    # autoPad is NOTSET -> use explicit padding

    if all(val == 0 for val in oPads):
        # No padding in any dieraction
        return tflPad.Padding.VALID

    if not __isSAMEPadding(oPads, oKernelShape):
        err.warning(f"TFLite does NOT support '{oPads}' padding for kernel '{oKernelShape}'! Using 'SAME'.")
    
    return tflPad.Padding.SAME





def convertNode(oNode: onnxN.Node, tensorForName: Callable[[str],tflT.Tensor]) -> tflO.Operator:
    """ Create a TFLite 'Operator' from the ONNX 'Node' with corresponding 'inputs' and 'outputs'.
        'tensorForName' is a function that maps the name of a tensor to the Tensor. """
    tOperator = tflO.Operator()

    # Initialize operator inputs
    tOperator.inputs = tflO.Inputs([])
    for name in oNode.inputs:
        tOperator.tmpInputs.append(tensorForName(name))

    # Initialize operator outputs
    tOperator.outputs = tflO.Outputs([])
    for name in oNode.outputs:
        tOperator.tmpOutputs.append(tensorForName(name))

    return tOperator




""" -------------------- Operator Conversion -------------------- 
    The following functions take an ONNX operator and convert it to its TFLite equivalent (if possible).
    They retrun a TUPLE! The first value is the equivalent TFLite operator.
    The second return value is the 'BuiltinOperator' code, for the generated operator. """



def convertConv(oConv: onnxConv.Conv) -> tuple[tflMeta.BuiltinOptions, tflBO.BuiltinOperator]:
    """ Convert the ONNX 'Conv' operator to TFLite. """

    match len(oConv.kernelShape):
        case 2:
            # 2D Convolution

            tConv = tflConv2D.Conv2D()

            __assign2DStrides(tConv, oConv.strides)

            if __isOfSize(oConv.dilations, 2):
                tConv.dilationHFactor = oConv.dilations[0]
                tConv.dilationHFactor = oConv.dilations[1]

            tConv.padding = __convertPadding(oConv.autoPad, oConv.pads, oConv.kernelShape)

            # TODO tConv.fusedActivationFunction

            return tConv, tflBO.BuiltinOperator.CONV_2D

        case 3:
            err.error("Conv3D NEEDS to be implemented and converted!")
        case _:
            err.error(f"Convolution with kernel shape '{oConv.kernelShape}' is not supported!")

def convertLRN(oLRN: onnxLRN.LRN) -> tuple[tflMeta.BuiltinOptions, tflBO.BuiltinOperator]:
    """ Convert ONNX 'LRN' to TFLite 'LocalResponseNormalization'. """

    tLRN = tflLRN.LRN()

    tLRN.radius = oLRN.size // 2 # TODO Investigate conversion
    tLRN.bias = oLRN.bias
    tLRN.alpha = oLRN.alpha
    tLRN.beta = oLRN.beta

    return tLRN, tflBO.BuiltinOperator.LOCAL_RESPONSE_NORMALIZATION


def convertMaxPool(oMP: onnxMaxPool.MaxPool) -> tuple[tflMeta.BuiltinOptions, tflBO.BuiltinOperator]:
    """ Convert the ONNX 'MaxPool' operator to TFLite 'MaxPool2D'/ """

    match len(oMP.kernelShape):
        case 2:
            # 2D MaxPool

            tMP = tflMaxPool2D.MaxPool2D()

            __assign2DStrides(tMP, oMP.strides)

            if __isOfSize(oMP.kernelShape, 2):
                tMP.filterH = oMP.kernelShape[0]
                tMP.filterW = oMP.kernelShape[1]

            tMP.padding = __convertPadding(oMP.autoPad, oMP.pads, oMP.kernelShape)

            # TODO tMP.fusedActivationFunction

            if oMP.dilations is not None:
                err.note("MaxPool dilations cannot be converted to TFLite!")

            return tMP, tflBO.BuiltinOperator.MAX_POOL_2D

        case _:
            err.error(f"MaxPool with kernel shape '{oMP.kernelShape}' is not supported!")

def convertReshape(oNode: onnxN.Node, 
        tflBufferForName: Callable[[str],tflB.Buffer]) -> tuple[tflMeta.BuiltinOptions, tflBO.BuiltinOperator]:
    """ Convert ONNX 'Reshape' to TFLite 'Reshape'. """

    buffer = tflBufferForName(oNode.inputs[1])

    if buffer is None:
        err.error(err.Code.INVALID_ONNX_OPERATOR, "ONNX Reshape did NOT have a 'shape' input tensor.")

    tReshape = tflReshape.Reshape(buffer.data.tolist())

    return tReshape, tflBO.BuiltinOperator.RESHAPE
