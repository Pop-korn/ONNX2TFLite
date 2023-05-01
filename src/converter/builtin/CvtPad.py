"""
    CvtPad

Convert ONNX operator Pad to TFLite.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

import src.err as err

import src.parser.builtin.Pad as onnxPad

import src.generator.meta.meta as tflMeta

def convert(oPad: onnxPad.Pad) -> tflMeta.BuiltinOptions:
    """ Convert ONNX operator 'Pad' to TFLite """
    # Currently only skipping this operator, when its attributes are zeroes
    # is supported
    try:

        if oPad.mode == b"constant" and all([ el == 0 for el in oPad.pads ]):
            # Operator can be skipped
            return None
        else:
            err.error(err.Code.NOT_IMPLEMENTED, "Conversion of operator 'Pad' with"
                    f"attributes {oPad.mode}, {oPad.pads} and {oPad.value}",
                    "is not yet supported!")
            
    except Exception as e:
        err.error(err.Code.UNSUPPORTED_OPERATOR_ATTRIBUTES,
                  "Conversion of ONNX operator 'Pad' is limited and given model",
                  "contains unsupported attribute/input combinations!")
        
