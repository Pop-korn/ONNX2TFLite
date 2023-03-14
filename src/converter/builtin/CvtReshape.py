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

        if len(tflOperator.tmpInputs) != 2:
            err.warning("ONNX: Reshape operator doesn't have 2 inputs!",
                        "Conversion behaviour is undefined!")
            
        if not modelBuilder.tensorHasData(tflOperator.tmpInputs[1]):
            # The new shape is dynamically calculated during inference.
            # Currently this is not supported. Just return an empty 'Reshape'.
            return tflReshape.Reshape([])


        """ The new shape is stacically given. """

        buffer = tflOperator.tmpInputs[1].tmpBuffer

        newShape = buffer.data.tolist()

        tReshape = tflReshape.Reshape(newShape)

        # The new shape can be represented using operators parameters. No need
        # for the input tensor. Remove it
        tflOperator.tmpInputs.pop()

        originalShape = tflOperator.tmpInputs[0].shape.vector
        if not Translator.isNCHW(newShape):
            if len(originalShape) > len(newShape):
                """ This operator is used to 'flatten' the input tensor.
                    If equivalent NCHW and NHWC tensors are flattened,
                    the result will be different! So input first needs to be
                    converted to NCHW. """
                tflOperator.tmpInputs[0] = modelBuilder.nchwVersionOf(tflOperator.tmpInputs[0])

        return tReshape
    
    except Exception as e:
        print(e)
        err.error(err.Code.INVALID_ONNX_OPERATOR, 
                  "ONNX Reshape conversion failed!")
        