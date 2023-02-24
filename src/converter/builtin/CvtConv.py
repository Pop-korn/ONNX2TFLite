import src.converter.conversion.Translator as Translator
import src.converter.conversion.common as common

import src.err as err

import src.parser.builtin.Conv as onnxConv

import src.generator.builtin.Conv2D as tflConv2D
import src.generator.meta.meta as tflMeta

import lib.tflite.BuiltinOperator as tflBO

def convert(oConv: onnxConv.Conv) -> tuple[tflMeta.BuiltinOptions, 
                                           tflBO.BuiltinOperator]:
    """ Convert the ONNX 'Conv' operator to TFLite. """

    match len(oConv.kernelShape):
        case 2:
            # 2D Convolution
            tConv = tflConv2D.Conv2D()

            common.assign2DStrides(tConv, oConv.strides)

            if common.isOfSize(oConv.dilations, 2):
                tConv.dilationHFactor = oConv.dilations[0]
                tConv.dilationHFactor = oConv.dilations[1]

            tConv.padding = Translator.convertPadding(oConv.autoPad, oConv.pads, 
                                                      oConv.kernelShape)
            
            # TODO tConv.fusedActivationFunction
            
            return tConv, tflBO.BuiltinOperator.CONV_2D
        
        case 3:
            err.error("Conv3D NEEDS to be implemented and converted!")
        case _:
            err.error(f"Convolution with kernel shape '{oConv.kernelShape}'",
                      "is not supported!")
