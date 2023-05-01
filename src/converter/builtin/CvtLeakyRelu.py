"""
    CvtLeakyRelu

Convert ONNX operator LeakyRelu to TFLite LeakyRelu.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

import src.parser.builtin.LeakyRelu as onnxLR

import src.generator.builtin.LeakyRelu as tflLR

def convert(oLR: onnxLR.LeakyRelu) -> tflLR.LeakyRelu:
    return tflLR.LeakyRelu(oLR.alpha)
