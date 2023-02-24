from typing import Callable

import src.converter.builder.ModelBuilder as ModelBuilder

from src.converter.builtin import CvtConv, CvtLRN, CvtMaxPool, CvtReshape

import src.generator.model.Operators as tflO

import src.parser.model.Nodes as onnxN

import src.err as err

import lib.tflite.BuiltinOperator as tflBO


class OperatorConverter:
    """ This class provides methods to convert ONNX operators to TFLite and 
        create them using the provided 'ModelBuilder'. """

    __builder: ModelBuilder.ModelBuilder

    def __init__(self, builder: ModelBuilder.ModelBuilder) -> None:
        self.__builder = builder


    def convertNode(self, oNode: onnxN.Node) -> tflO.Operator:
        """ Create a TFLite 'Operator' from the ONNX 'Node' with corresponding 
            'inputs' and 'outputs'. """
        
        tOperator = tflO.Operator()

        # Initialize operator inputs
        tOperator.inputs = tflO.Inputs([])
        for name in oNode.inputs:
            tOperator.tmpInputs.append(self.__builder.tensorForName(name))

        # Initialize operator outputs
        tOperator.outputs = tflO.Outputs([])
        for name in oNode.outputs:
            tOperator.tmpOutputs.append(self.__builder.tensorForName(name))

        return tOperator


    def convertOperator(self, oNode:onnxN.Node):
        """ Convert an ONNX Node to a corresponding TFLite operator.
            This is ALWAYS a 1 to 1 conversion. """

        tOp = self.convertNode(oNode)

        match(oNode.opType):
            case "Conv":
                tOp.builtinOptions, opCode = CvtConv.convert(oNode.attributes)
                tOp.opcodeIndex = self.__builder.opCodeIndexForOpType(opCode)
            case "LRN":
                tOp.builtinOptions, opCode = CvtLRN.convert(oNode.attributes)
                tOp.opcodeIndex = self.__builder.opCodeIndexForOpType(opCode)
            case "MaxPool":
                tOp.builtinOptions, opCode = CvtMaxPool.convert(oNode.attributes)
                tOp.opcodeIndex = self.__builder.opCodeIndexForOpType(opCode)
            case "Relu":
                tOp.builtinOptions = None
                tOp.opcodeIndex = self.__builder.opCodeIndexForOpType(
                                                    tflBO.BuiltinOperator.RELU)
            case "Reshape":
                tOp.builtinOptions, opCode = CvtReshape.convert(tOp)
                tOp.opcodeIndex = self.__builder.opCodeIndexForOpType(opCode)
            case _:
                err.error(None, f"Conversion of ONNX Operator '{oNode.opType}'",
                          "is not yet supported!")

        self.__builder.getOperators().append(tOp)
