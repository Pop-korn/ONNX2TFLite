import src.generator.model.Model as tflM

import src.parser.model.Model as onnxM

import src.convertor.builder.ModelBuilder as ModelBuilder
import src.convertor.conversion.OperatorConverter as OperatorConverter


def convertModel(oM: onnxM.Model) -> tflM.Model:
    description="doc:'"+oM.docString+f"' domain:'"+oM.domain+"' producer:'"+oM.producerName+" "+oM.producerVersion+"'"

    builder = ModelBuilder.ModelBuilder(oM.modelVersion, description)

    builder.buildOutputTensors(oM.graph.outputs)
    builder.buildInputTensors(oM.graph.inputs)

    builder.buildConstantTensors(oM.graph.initializers)

    builder.buildInternalTensors(oM.graph.valueInfo)

    opConvert = OperatorConverter.OperatorConverter(builder)


    for oNode in oM.graph.nodes:
        opConvert.buildOperator(oNode)

    return builder.finish()


