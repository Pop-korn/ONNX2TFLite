"""
    protobufFromJson

Simple script to convert an ONNX (Protocol Buffer) file to JSON format.
Takes one argument, which is the input JSON file.


__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

from google.protobuf.json_format import MessageToJson

import onnx

import sys

if len(sys.argv) != 2:
    print("Require 1 .onnx file!")
    exit(1)

file = sys.argv[1]

model = onnx.load(file)

json = MessageToJson(model)

# Shorten long lines
# json = [line if len(line) < 100 else line[0:50] for line in json.split("\n")]
# json = "\n".join(json)

print(json)
