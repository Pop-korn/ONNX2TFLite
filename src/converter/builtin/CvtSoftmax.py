import src.err as err

import src.parser.builtin.Softmax as onnxSoftmax

import src.generator.builtin.Softmax as tflSoftmax

def convert(oSM: onnxSoftmax.Softmax) -> tflSoftmax.Softmax:
    if oSM.axis != -1:
        err.warning("ONNX pperator 'Softmax' has attribute 'axis' = ",
                    f"'{oSM.axis}'. Only '-1' can be converted to TFLite!")

    tSM = tflSoftmax.Softmax(1.0) # TODO Check validity

    return tSM
