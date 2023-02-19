import src.generator.model.Model as tflM

import src.parser.model.Model as onnxM

import src.convertor.ModelBuilder as ModelBuilder

def convertModel(oM: onnxM.Model) -> tflM.Model:

    builder = ModelBuilder.Builder(oM.modelVersion)

    builder.buildOutputTensors(oM.graph.outputs)
    
    return builder.finish()


