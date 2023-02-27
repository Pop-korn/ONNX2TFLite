import src.converter.conversion.Translator as Translator
import src.converter.conversion.common as common

import src.err as err

import lib.tflite.BuiltinOperator as tflBO

import src.parser.builtin.MaxPool as onnxMaxPool

import src.generator.builtin.MaxPool2D as tflMaxPool2D
import src.generator.meta.meta as tflMeta


def convert(oMP: onnxMaxPool.MaxPool) -> tflMeta.BuiltinOptions:
    """ Convert the ONNX 'MaxPool' operator to TFLite 'MaxPool2D'. """

    match len(oMP.kernelShape):
        case 2:
            # 2D MaxPool

            tMP = tflMaxPool2D.MaxPool2D()

            common.assign2DStrides(tMP, oMP.strides)

            if common.isOfSize(oMP.kernelShape, 2):
                tMP.filterH = oMP.kernelShape[0]
                tMP.filterW = oMP.kernelShape[1]

            tMP.padding = Translator.convertPadding(oMP.autoPad, oMP.pads, 
                                                    oMP.kernelShape)

            # TODO tMP.fusedActivationFunction

            if oMP.dilations is not None:
                err.warning("MaxPool dilations cannot be converted to TFLite!")

            return tMP

        case _:
            err.error(f"MaxPool with kernel shape '{oMP.kernelShape}'",
                      "is not supported!")
