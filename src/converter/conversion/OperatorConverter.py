"""
    OperatorConverter

Module contains high level functions to convert ONNX operators to TFLite.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

import src.converter.builder.ModelBuilder as ModelBuilder


from src.converter.builtin import (
    CvtConv, CvtLRN, CvtMaxPool, CvtReshape, CvtDropout, CvtSoftmax, CvtGemm, 
    CvtMul, CvtAdd, CvtBatchNormalization, CvtLeakyRelu, CvtSum, CvtPad,
    CvtAveragePool, CvtTranspose, CvtLogSoftmax, CvtConstant
)


import src.generator.model.Operators as tflO
import src.converter.builtin.groups.Cvt_MatMul_Add as Cvt_MatMul_Add

import src.parser.model.Nodes as onnxN

import src.err as err

import lib.tflite.BuiltinOperator as tflBO

from typing import List


class OperatorConverter:
    """ This class provides methods to convert ONNX operators to TFLite and 
        create them using the provided 'ModelBuilder'. """

    __builder: ModelBuilder.ModelBuilder

    def __init__(self, builder: ModelBuilder.ModelBuilder) -> None:
        self.__builder = builder


    def __createOperator(self, opType: tflBO.BuiltinOperator, 
                         inputs: List[str], outputs: List[str]) -> tflO.Operator:
        """ Create a TFLite operator with given operator type, inputs and 
            outputs and return it. """
        
        tOp = tflO.Operator(
            opcodeIndex=self.__builder.opCodeIndexForOpType(opType),
            inputs=tflO.Inputs(),
            outputs=tflO.Outputs()
        )

        for name in inputs:
            tOp.tmpInputs.append(self.__builder.tensorForName(name))
        for name in outputs:
            tOp.tmpOutputs.append(self.__builder.tensorForName(name))

        return tOp


    def __convertNode(self, oNode: onnxN.Node) -> tflO.Operator:
        """ Create a TFLite 'Operator' from the ONNX 'Node' with corresponding 
            'inputs' and 'outputs'. """
        
        tOperator = tflO.Operator()

        # Initialize operator inputs
        tOperator.inputs = tflO.Inputs()
        for name in oNode.inputs:
            tOperator.tmpInputs.append(self.__builder.tensorForName(name))

        # Initialize operator outputs
        tOperator.outputs = tflO.Outputs()
        for name in oNode.outputs:
            tOperator.tmpOutputs.append(self.__builder.tensorForName(name))

        return tOperator


    def convertOperator(self, oNode:onnxN.Node):
        """ Convert an ONNX Node to a corresponding TFLite operator.
            This is ALWAYS a 1 to 1 conversion. """

        tOp = self.__convertNode(oNode)

        # Indicator if after conversion, 'tOp.builtinOptions' was set
        implicitOperatorType = True

        # Identify ONNX operator and convert it
        if oNode.opType == "Gemm":
            tOp.builtinOptions = CvtGemm.convert(oNode.attributes,
                                                 tOp,
                                                 self.__builder)
        elif oNode.opType == "LeakyRelu":
            tOp.builtinOptions = CvtLeakyRelu.convert(oNode.attributes)
        elif oNode.opType == "LogSoftmax":
            tOp.builtinOptions = CvtLogSoftmax.convert(oNode.attributes, tOp)
        elif oNode.opType == "LRN":
            tOp.builtinOptions = CvtLRN.convert(oNode.attributes)
        elif oNode.opType == "Mul":
            tOp.builtinOptions = CvtMul.convert()
        elif oNode.opType == "Relu":
            tOp.builtinOptions = None
            tOp.opcodeIndex = self.__builder.opCodeIndexForOpType(tflBO.BuiltinOperator.RELU)
            implicitOperatorType = False
        elif oNode.opType == "Reshape":
            tOp.builtinOptions = CvtReshape.convert(tOp, self.__builder)
        elif oNode.opType == "Sum":
            tOp.builtinOptions = CvtSum.convert(tOp)
        elif oNode.opType == "Transpose":
            tOp.builtinOptions = CvtTranspose.convert(oNode.attributes,
                                                      tOp, self.__builder)


            """ Operators that might not get converted! """
        elif oNode.opType == "Add":
            tOp.builtinOptions = CvtAdd.convert(tOp, self.__builder)
            if tOp.builtinOptions is None:
                self.__builder.skipOperator(tOp)
                return
        elif oNode.opType == "Constant":
            tOp.builtinOptions = CvtConstant.convert(oNode.attributes,
                                                     self.__builder)
            if tOp.builtinOptions is None:
                return
        elif oNode.opType == "Dropout":
            tOp.builtinOptions = CvtDropout.convert(oNode.attributes)
            if tOp.builtinOptions is None:
                self.__builder.skipOperator(tOp)
                return
        elif oNode.opType == "Pad":
            tOp.builtinOptions = CvtPad.convert(oNode.attributes)
            if tOp.builtinOptions is None:
                self.__builder.skipOperator(tOp)
                return


            """ Operators that handle adding operators to the model themselves """
        elif oNode.opType == "AveragePool":
            CvtAveragePool.convert(oNode.attributes, tOp, self.__builder)
            return
        elif oNode.opType == "BatchNormalization":
            CvtBatchNormalization.convert(oNode.attributes, tOp,
                                          self.__builder)
            return
        elif oNode.opType == "Conv":
            CvtConv.convert(oNode.attributes, tOp, self.__builder)
            return
        elif oNode.opType == "MaxPool":
            CvtMaxPool.convert(oNode.attributes, tOp, self.__builder)
            return
        elif oNode.opType == "Softmax":
            CvtSoftmax.convert(oNode.attributes, tOp, self.__builder)
            return

        else:
            implicitOperatorType = False
            err.error(err.Code.UNSUPPORTED_OPERATOR,
                      f"Conversion of ONNX Operator '{oNode.opType}'",
                      "is not yet supported!")
            return
                
        # Assign 'tOp' its operator type. If possible
        if implicitOperatorType:
            tOp.opcodeIndex = self.__builder.opCodeIndexForOpType(tOp.builtinOptions.operatorType)
        

        self.__builder.checkAndAppendOperator(tOp)


    def convert_MatMul_Add(self, matMul: onnxN.Node, add: onnxN.Node):
        """ Convert ONNX MatMul and Add operators to TFLite FullyConnected. """

        # New operator inputs
        inputs = [ name for name in matMul.inputs ]
        # Add the input of 'Add' which is not the output of 'MatMul'
        if add.inputs[0] == matMul.outputs[0]:
            inputs.append(add.inputs[1])
        else:
            inputs.append(add.inputs[0])

        # New operator outputs
        outputs = [ name for name in add.outputs ]


        # Create the operator
        tOp = self.__createOperator(tflBO.BuiltinOperator.FULLY_CONNECTED,
                                    inputs, outputs)

        # Adjust the input tensor shapes to be compatible and add the operator
        Cvt_MatMul_Add.convert(tOp, self.__builder)
