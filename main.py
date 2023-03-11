import src.parser.model.Model as m

import src.converter.convert as convert

import flatbuffers as fb

def convertModel(onnxFile, tfliteFile):
    onnxModel = m.Model(onnxFile)

    tflModel = convert.convertModel(onnxModel)

    fbB = fb.Builder(2000000000)

    tflModel.genTFLite(fbB)

    buffer = fbB.Output()

    with open(tfliteFile, "wb") as f:
        f.write(buffer)

# convertModel("data/onnx/bvlcalexnet-12.onnx", "test/alexnet.tflite")
