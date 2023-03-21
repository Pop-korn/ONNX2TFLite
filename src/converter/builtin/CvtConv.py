import src.converter.conversion.Translator as Translator
import src.converter.conversion.common as common
import src.converter.builder.ModelBuilder as ModelBuilder

import src.err as err

import src.parser.builtin.Conv as onnxConv

import src.generator.builtin.Conv2D as tflConv2D
import src.generator.meta.meta as tflMeta
import src.generator.model.Operators as tflO

def convert(oConv: onnxConv.Conv, tOp: tflO.Operator,
            modelBuilder: ModelBuilder.ModelBuilder) -> tflMeta.BuiltinOptions:
    """ Convert the ONNX 'Conv' operator to TFLite. """

    match len(oConv.kernelShape):
        case 2:
            # 2D Convolution
            tConv = tflConv2D.Conv2D()

            common.assign2DStrides(tConv, oConv.strides)

            if common.isOfSize(oConv.dilations, 2):
                tConv.dilationHFactor = oConv.dilations[0]
                tConv.dilationWFactor = oConv.dilations[1]

            tConv.padding = Translator.convertPadding(oConv.autoPad, oConv.pads, 
                                                      oConv.kernelShape,
                                                      oConv.dilations)
            
            if len(tOp.tmpInputs) == 2:
                # Operator is has no bias. ONNX model can ommit it. TFLite can't.
                kernelShape = tOp.tmpInputs[1].shape.vector
                bias = modelBuilder.createZerosTensor([kernelShape[0]], 
                                                      "zero_conv_bias",
                                                      tOp.tmpInputs[1].tmpBuffer.data.dtype,
                                                      True)
                tOp.tmpInputs.append(bias)

            return tConv
        
        case 3:
            err.error("Conv3D NEEDS to be implemented and converted!")
        case _:
            err.error(f"Convolution with kernel shape '{oConv.kernelShape}'",
                      "is not supported!")
