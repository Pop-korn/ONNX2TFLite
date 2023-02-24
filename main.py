import src.parser.model.Model as m

import src.converter.convert as convert

import flatbuffers as fb

onnxModel = m.Model("data/onnx/bvlcalexnet-12.onnx")

tflModel = convert.convertModel(onnxModel)

fbB = fb.Builder(2000000000)

tflModel.genTFLite(fbB)

buffer = fbB.Output()

with open("test/alexnet.tflite","wb") as f:
    f.write(buffer)
