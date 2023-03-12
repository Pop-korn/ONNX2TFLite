import src.err as err

import src.generator.builtin.Reshape as tflReshape
import src.generator.meta.meta as tflMeta
import src.generator.model.Operators as tflOperators

import src.converter.conversion.Translator as Translator
import src.converter.builder.ModelBuilder as ModelBuilder

def convert(tflOperator: tflOperators.Operator,
            modelBuilder: ModelBuilder.ModelBuilder) -> tflMeta.BuiltinOptions:
    """ Convert ONNX 'Reshape' to TFLite 'Reshape'. """

    try:    
        buffer = tflOperator.tmpInputs[1].tmpBuffer

        newShape = buffer.data.tolist()

        tReshape = tflReshape.Reshape(newShape)

        # The input tensor was retained from the ONNX model. 
        # TFLite does NOT use it -> remove it
        tflOperator.tmpInputs.pop()

        originalShape = tflOperator.tmpInputs[0].shape.vector
        if not Translator.isNCHW(newShape):
            if len(originalShape) > len(newShape):
                """ This operator is used to 'flatten' the input tensor.
                    If equivalent NCHW and NHWC tensors are flattened,
                    the result will be different! So input first needs to be
                    converted to NCHW. """
                # TODO Prepend operator, to transform input
                pass

        return tReshape
    
    except:
        err.error(err.Code.INVALID_ONNX_OPERATOR, 
                  "ONNX Reshape did NOT have a 'shape' input tensor.")
        