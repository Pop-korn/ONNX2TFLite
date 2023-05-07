"""
    protobufFromJson

Simple script to convert a JSON file to the ONNX (Protocol Buffer) format.
Takes one argument, which is the input JSON file. Ouput is always 'test.onnx'.


__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

from google.protobuf.json_format import Parse

import onnx

import sys

if len(sys.argv) != 2:
    print("Require 1 .json file!")
    exit(1)

file = sys.argv[1]

with open(file,"r") as f:
    json = f.read()


with open("../data/schemas/onnx/onnx/onnx-ml.proto","r") as f:
    schema = f.read()

model = Parse(json, onnx.ModelProto())

onnx.save(model,"test.onnx")