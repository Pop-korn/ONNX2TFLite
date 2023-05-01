"""
    CvtDropout

Convert ONNX operator Dropout to TFLite.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

import src.parser.builtin.Dropout as onnxDropout

import src.generator.meta.meta as tflMeta

def convert(oDropout: onnxDropout.Dropout) -> tflMeta.BuiltinOptions | None:
    """ Convert the ONNX 'Dropout' operator to TFLite. 
        There is no direct equivalent.
        
        May return 'None' to indicate that the operator is to be skipped!
    """
    return None