import src.parser.model.Model as m

import src.convertor.convert as convert

import flatbuffers as fb

onnxModel = m.Model("data/onnx/bvlcalexnet-12.onnx")

tflModel = convert.convertModel(onnxModel)

fbB = fb.Builder()

tflModel.genTFLite(fbB)

buffer = fbB.Output()

with open("test/alexnet.tflite","wb") as f:
    f.write(buffer)
