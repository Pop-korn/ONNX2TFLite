"""
    convert

Entry point for ONNX to TFLite model conversion.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""


import src.generator.model.Model as tflM

from src.parser.model import Model as onnxM, Nodes as onnxN, Tensors as onnxT

import src.converter.builder.ModelBuilder as ModelBuilder
from src.converter.conversion import OperatorConverter, TensorConverter

import flatbuffers as fb


def __nodeAtIdxIsType(oNodes: onnxN.Nodes, idx: int, opType: str) -> bool:
    """ Determine if node in 'oNodes' at index 'idx' has type 'opType', if such
        node even exists. """
    try:

        return oNodes[idx].opType == opType
    except:
        return False
    

def __tensorIsOnlyUsedOnce(oNodes: onnxN.Nodes, name: str, idx: int):
    """ Determine if tensor with 'name' is the input of exactly 1 operator in 
        'oNodes'. 'idx' is the index of the operator that produces the tensor
        as its output.
        """
    refCnt = 0

    for oNode in oNodes[idx:]:
        if name in oNode.inputs:
            refCnt += 1
            if refCnt > 1:
                return False
            
    return refCnt == 1


def __convertOperators(oNodes: onnxN.Nodes, 
                       operatorCvt: OperatorConverter.OperatorConverter):
    """ Find the best way to convert all operators in the ONNX model and
        convert them to TFLite. """
    
    opsToSkip = 0

    for idx, oNode in enumerate(oNodes):

        if opsToSkip > 0:
            # Skip operators if needed
            opsToSkip -= 1
            continue

        if oNode.opType == "MatMul":
            if __nodeAtIdxIsType(oNodes, idx+1, "Add"):

                if __tensorIsOnlyUsedOnce(oNodes, oNode.inputs[0], idx):
                    # MatMul operator followed by Add operator -> convert to 
                    # FullyConnected
                    operatorCvt.convert_MatMul_Add(oNode, oNodes[idx + 1])
                    
                    # Skip the Add operator
                    opsToSkip = 1
                    continue


        operatorCvt.convertOperator(oNode)



def __convert(oM: onnxM.Model) -> tflM.Model:
    description="doc:'"+oM.docString+f"' domain:'"+oM.domain+"' producer:'"+oM.producerName+" "+oM.producerVersion+"'"

    builder = ModelBuilder.ModelBuilder(3, description)
    operatorCvt = OperatorConverter.OperatorConverter(builder)
    tensorCvt = TensorConverter.TensorConverter(builder)
    
    tensorCvt.convertOutputTensors(oM.graph.outputs)
    tensorCvt.convertInputTensors(oM.graph.inputs)

    tensorCvt.convertConstantTensors(oM.graph.initializers)

    tensorCvt.convertInternalTensors(oM.graph.valueInfo)

    __convertOperators(oM.graph.nodes, operatorCvt)

    return builder.finish()


def convertModel(onnxFile, tfliteFile):
    """ Convert an ONNX model stored in 'onnxFile' to TFLite.
        Store the resulting TFLite model in 'tfliteFile'. """
    
    onnxModel = onnxM.Model(onnxFile)

    tflModel = __convert(onnxModel)

    fbB = fb.Builder()

    tflModel.genTFLite(fbB)

    buffer = fbB.Output()

    with open(tfliteFile, "wb") as f:
        f.write(buffer)
