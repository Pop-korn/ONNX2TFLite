import src.err as err

import src.parser.builtin.LRN as onnxLRN

import src.generator.builtin.LRN as tflLRN
import src.generator.meta.meta as tflMeta

def convert(oLRN: onnxLRN.LRN) -> tflMeta.BuiltinOptions:
    """ Convert ONNX 'LRN' to TFLite 'LocalResponseNormalization'. """

    tLRN = tflLRN.LRN()

    if tLRN.radius % 2 != 0:
        err.internal("ONNX LRN operator has even radius ('{tLRN.radius}')!",
                     "This is not expected.")

    tLRN.radius = oLRN.size // 2 # TODO Investigate conversion
    tLRN.bias = oLRN.bias
    tLRN.alpha = oLRN.alpha
    tLRN.beta = oLRN.beta

    return tLRN
