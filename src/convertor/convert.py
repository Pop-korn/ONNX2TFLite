import src.generator.model.Model as tflM

import src.parser.model.Model as onnxM

import src.convertor.ModelBuilder as ModelBuilder

def convertModel(oM: onnxM.Model) -> tflM.Model:
    description="doc:'"+oM.docString+f"' domain:'"+oM.domain+"' producer:'"+oM.producerName+" "+oM.producerVersion+"'"

    builder = ModelBuilder.Builder(oM.modelVersion, description)

    builder.buildOutputTensors(oM.graph.outputs)
    builder.buildInputTensors(oM.graph.inputs)

    builder.buildConstantTensors(oM.graph.initializers[1:2])

    return builder.finish()


