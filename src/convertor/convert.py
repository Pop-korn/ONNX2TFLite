import src.generator.model.Model as tflM

import src.parser.model.Model as onnxM

import src.convertor.ModelBuilder as ModelBuilder

import flatbuffers as fb

def convertModel(oM: onnxM.Model) -> tflM.Model:

    builder = ModelBuilder.Builder(oM.modelVersion)

    builder.buildOutputTensors(oM.graph.outputs)

    
    tflModel = builder.finish()

    fbB = fb.Builder()
    tflModel.genTFLite(fbB)

    genModelFile = "test/alexnet.tflite"
    buffer = fbB.Output()
    with open(genModelFile,"wb") as f:
        f.write(buffer)

