import src.converter.convert as convert
import src.err as err

import argparse
import ntpath


""" Create argument parser """
parser = argparse.ArgumentParser(
    prog = "onnx2tflite",
    description = """
        Convert a '.onnx' DNN model to an equivalent '.tflite' model.
        By default the output '.tflite' file will be generated in the current 
        working directory and have the same name as the input '.onnx' file.
    """,
    usage="python onnx2tflite.py <input_file.onnx> [-o/--output <output_file.tflite> --verbose]"
)

parser.add_argument("onnxFile")
parser.add_argument("-o", "--output", type=str, required=False, metavar="out", help="output '.tflite' file")
parser.add_argument("--verbose", action="store_true", help="print detailed internal messages")


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

print(f"Succesfully converted '{inputOnnxFile}' model to '{outputTFLiteFile}'.")
