import src.converter.convert as convert
import src.err as err

import sys

if len(sys.argv) < 3:
    err.error(err.Code.INPUT_ERR, "You need to specify the imput .onnx file and",
              "the output .tflite file.",
              "Example use: 'python onnx2tflite model.onnx out/model.tflite'.")


if len(sys.argv) >= 4:
    errLevel = int(sys.argv[3])
    err.MIN_OUTPUT_IMPORTANCE = err.MessageImportance(errLevel)


convert.convertModel(sys.argv[1], sys.argv[2])
