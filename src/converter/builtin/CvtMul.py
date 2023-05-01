"""
    CvtMul

Convert ONNX operator Mul to TFLite Mul.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

import src.generator.builtin.Mul as tflMul

def convert() -> tflMul.Mul:
    return tflMul.Mul()
