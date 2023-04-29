import src.converter.convert as convert
import src.err as err

import sys

import argparse
import ntpath


""" Create argument parser """
parser = argparse.ArgumentParser(
    prog = "onnx2tflite",
    description = "Convert a '.onnx' DNN model to an equivalent '.tflite' model.",
    usage="python onnx2tflite <input_file.onnx> [-o/--output <output_file.tflite> --verbose]",
    epilog = "Koniec"
)

parser.add_argument("onnxFile")
parser.add_argument("-o", "--output", type=str, required=True, metavar="output_tflite_file")
parser.add_argument("--verbose", action="store_true")


""" Parse program arguments """
args = parser.parse_args()
inputOnnxFile = args.onnxFile
outputTFLiteFile = args.output

if outputTFLiteFile is None:
    # Default output file has the same name as input, but with a different file 
    # extension and is in the current working directory

    fileName = ntpath.basename(inputOnnxFile) # Get filename
    fileName = ntpath.splitext(fileName)[0] # Remove '.onnx' extension
    outputTFLiteFile = fileName + ".tflite"

if args.verbose:
    # Print all logging messages
    err.MIN_OUTPUT_IMPORTANCE = err.MessageImportance.LOWEST



""" Convert the model """
convert.convertModel(inputOnnxFile, outputTFLiteFile)

print(inputOnnxFile, outputTFLiteFile)
