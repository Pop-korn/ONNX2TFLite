import src.err as err

import src.parser.builtin.LRN as onnxLRN

import src.generator.builtin.LRN as tflLRN
import src.generator.meta.meta as tflMeta


def convert(oLRN: onnxLRN.LRN) -> tflMeta.BuiltinOptions:
    """ Convert ONNX 'LRN' to TFLite 'LocalResponseNormalization'. """

    tLRN = tflLRN.LRN()

    if oLRN.size % 2 == 0:
        err.warning(f"ONNX: LRN Operator has even size ({oLRN.size}).",
                    "So the generated TFLite model might NOT be identical!")
        # ONNX inference uses 'ciel(size/2)' and 'floor(size/2)'
        # TFLite inference only uses radius, so 'floor(size/2)'

    tLRN.radius = oLRN.size // 2
    tLRN.bias = oLRN.bias
    
    # Probably due to float accuracy, TFLite operator will give slightly
    # different output. Difference appears to be on the scale of 10^(-4).
    tLRN.alpha = oLRN.alpha / oLRN.size 
    
    tLRN.beta = oLRN.beta

    return tLRN
