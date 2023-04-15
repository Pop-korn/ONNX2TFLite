import src.err as err

import src.parser.builtin.LogSoftmax as onnxLogSoftmax

import src.generator.builtin.LogSoftmax as tflLogSoftmax
import src.generator.model.Operators as tflO

def convert(oLSM: onnxLogSoftmax.LogSoftmax,
            tOp: tflO.Operator) -> tflLogSoftmax.LogSoftmax:
    """ Convert ONNX 'LogSoftmax' to TFLite and return the options object. """

    if oLSM.axis != -1 and oLSM.axis != len(tOp.tmpInputs[0].shape.vector)-1:
        err.error(err.Code.NOT_IMPLEMENTED, "ONNX operator 'LogSoftmax' has",
                  f"attribute 'axis' = '{oLSM.axis}',",
                  f"with imput shape = {tOp.tmpInputs[0].shape.vector}.",
                  "Conversion of this case is not yet implemented!")
        
    return tflLogSoftmax.LogSoftmax()