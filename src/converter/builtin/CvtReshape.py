import src.err as err

import lib.tflite.BuiltinOperator as tflBO

import src.generator.builtin.Reshape as tflReshape
import src.generator.meta.meta as tflMeta
import src.generator.model.Operators as tflOperators

def convert(tflOperator: tflOperators.Operator) -> tuple[tflMeta.BuiltinOptions, 
                                                         tflBO.BuiltinOperator]:
    """ Convert ONNX 'Reshape' to TFLite 'Reshape'. """

    try:    
        buffer = tflOperator.tmpInputs[1].tmpBuffer

        tReshape = tflReshape.Reshape(buffer.data.tolist())

        # The input tensor was retained from the ONNX model. 
        # TFLite does NOT use it -> remove it
        tflOperator.tmpInputs.pop()

        return tReshape, tflBO.BuiltinOperator.RESHAPE
    
    except:
        err.error(err.Code.INVALID_ONNX_OPERATOR, 
                  "ONNX Reshape did NOT have a 'shape' input tensor.")
        