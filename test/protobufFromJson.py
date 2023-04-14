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