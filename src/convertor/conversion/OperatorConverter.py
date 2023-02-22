from typing import Callable

import src.generator.model.Operators as tflO

import src.generator.builtin.Conv2D as tflConv2D
import src.generator.builtin.LRN as tflLRN

import src.parser.model.Nodes as onnxN

import src.parser.builtin.Conv as onnxConv
import src.parser.builtin.LRN as onnxLRN

import src.err as err

import lib.tflite.Padding as tflPad


""" -------------------- Helper Operator Functions -------------------- """


def __convertPadding(autoPad: str, oPads: list[int]) -> tflPad.Padding:
    """ Convert ONNX pads to TFLite padding. 
        'autoPad' is the ONNX attribute 'auto_pad' and 'oPads' is the ONNX attribute 'pads'. """

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

    err.note(f"TFLite does NOT support '{oPads}' padding! Using 'SAME'.")
    return tflPad.Padding.SAME




""" -------------------- Operator Conversion -------------------- """


def convertNode(oNode: onnxN.Node, tensorIndexForName: Callable[[str],int]) -> tflO.Operator:
    """ Create a TFLite 'Operator' from the ONNX 'Node' with corresponding 'inputs' and 'outputs'.
        'tensorIndexForName' is a function that maps the name of a tensor to its index in the
        TFLite 'tensors' vector. """
    tOp = tflO.Operator()

    tOp.inputs = tflO.Inputs([ tensorIndexForName(name) for name in oNode.inputs ])
    tOp.outputs = tflO.Outputs([ tensorIndexForName(name) for name in oNode.outputs ])

    return tOp


def convertConv(oConv: onnxConv.Conv) -> tflConv2D.Conv2D:
    """ Convert the ONNX 'Conv' operator to TFLite """

    match len(oConv.kernelShape):
        case 2:
            tConv = tflConv2D.Conv2D()

            if len(oConv.strides) == 2:
                tConv.strideH = oConv.strides[0]
                tConv.strideW = oConv.strides[1]

            if oConv.dilations is not None and len(oConv.dilations) == 2:
                tConv.dilationHFactor = oConv.dilations[0]
                tConv.dilationHFactor = oConv.dilations[1]

            tConv.padding = __convertPadding(oConv.autoPad, oConv.pads)

            # TODO tConv.fusedActivationFunction

            return tConv

        case 3:
            err.error("Conv3D NEEDS to be implemented and converted!")
        case _:
            err.error(f"Convolution with kernel shape '{oConv.kernelShape}' is not supported!")

def convertLRN(oLRN: onnxLRN.LRN) -> tflLRN.LRN:
    """ Convert ONNX 'LRN' to TFLite 'LocalResponseNormalization'. """

    tLRN = tflLRN.LRN()

    tLRN.radius = oLRN.size // 2 # TODO Investigate conversion
    tLRN.bias = oLRN.bias
    tLRN.alpha = oLRN.alpha
    tLRN.beta = oLRN.beta

    return tLRN