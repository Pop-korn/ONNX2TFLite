import src.generator.model.Model as tflM

import src.parser.model.Model as onnxM

import src.converter.builder.ModelBuilder as ModelBuilder
import src.converter.conversion.OperatorConverter as OperatorConverter
import src.converter.conversion.TensorConverter as TensorConverter


def convertModel(oM: onnxM.Model) -> tflM.Model:
    description="doc:'"+oM.docString+f"' domain:'"+oM.domain+"' producer:'"+oM.producerName+" "+oM.producerVersion+"'"

    builder = ModelBuilder.ModelBuilder(3, description)
    operatorCvt = OperatorConverter.OperatorConverter(builder)
    tensorCvt = TensorConverter.TensorConverter(builder)
    

    tensorCvt.convertOutputTensors(oM.graph.outputs)
    tensorCvt.convertInputTensors(oM.graph.inputs)

    tensorCvt.convertConstantTensors(oM.graph.initializers)

    tensorCvt.convertInternalTensors(oM.graph.valueInfo)



    for oNode in oM.graph.nodes:
        operatorCvt.convertOperator(oNode)

    return builder.finish()


