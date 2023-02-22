from typing import Callable
import src.generator.model.Operators as tflO

import src.generator.builtin.Conv2D as tflConv2D

import src.parser.model.Nodes as onnxN

import src.parser.builtin.Conv as onnxConv

import lib.tflite.Padding as tflPad

import src.err as err


""" -------------------- Helper Operator Functions -------------------- """


def __convertPadding(oPads: list[int]) -> tflPad.Padding:
    """ Convert ONNX pads to TFLite padding. """
    if all(val == 0 for val in oPads):
        # No padding in any dieraction
        return tflPad.Padding.VALID

    err.note(f"TFLite does NOT support '{oPads}' padding! Using 'SAME', i.e. use as much padding as needed.")
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
    """ Convert the  """
    tConv = tflConv2D.Conv2D()

    if len(oConv.strides) == 2:
        tConv.strideH = oConv.strides[0]
        tConv.strideW = oConv.strides[1]

    if oConv.dilations is not None and len(oConv.dilations) == 2:
        tConv.dilationHFactor = oConv.dilations[0]
        tConv.dilationHFactor = oConv.dilations[1]

    tConv.padding = __convertPadding(oConv.pads)

    print(tConv.builtinOptionsType)

    return tConv