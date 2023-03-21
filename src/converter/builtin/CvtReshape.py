import src.err as err

import src.generator.builtin.Reshape as tflReshape
import src.generator.meta.meta as tflMeta
import src.generator.model.Operators as tflOperators

import src.converter.conversion.Translator as Translator
import src.converter.builder.ModelBuilder as ModelBuilder

def convert(tOp: tflOperators.Operator,
            modelBuilder: ModelBuilder.ModelBuilder) -> tflMeta.BuiltinOptions:
    """ Convert ONNX 'Reshape' to TFLite 'Reshape'. """

    try:    

        if len(tOp.tmpInputs) != 2:
            err.warning("ONNX: Reshape operator doesn't have 2 inputs!",
                        "Conversion behaviour is undefined!")
            
        if not modelBuilder.tensorHasData(tOp.tmpInputs[1]):
            # The new shape is dynamically calculated during inference.
            # Currently this is not supported. Just return an empty 'Reshape'.
            return tflReshape.Reshape([])


        """ The new shape is stacically given. """

        buffer = tOp.tmpInputs[1].tmpBuffer

        newShape = buffer.data.tolist()

        # Replace 'zeroes' with 'ones'. Onnx sometimes uses 0 intead of 1 for shapes
        newShape = [ dim if dim != 0 else 1 for dim in newShape ]

        tReshape = tflReshape.Reshape(newShape)

        # The new shape can be represented using operators parameters. No need
        # for the input tensor. Remove it
        tOp.tmpInputs.pop()

        originalShape = tOp.tmpInputs[0].shape.vector
        if not Translator.isNCHW(newShape):
            if len(originalShape) > len(newShape):
                """ This operator is used to 'flatten' the input tensor.
                    If equivalent NCHW and NHWC tensors are flattened,
                    the result will be different! So input first needs to be
                    converted to NCHW. """
                tOp.tmpInputs[0] = modelBuilder.nchwVersionOf(tOp.tmpInputs[0])

        return tReshape
    
    except Exception as e:
        print(e)
        err.error(err.Code.INVALID_ONNX_OPERATOR, 
                  "ONNX Reshape conversion failed!")
        