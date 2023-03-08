import src.converter.builder.ModelBuilder as ModelBuilder

from src.converter.builtin import CvtConv, CvtLRN, CvtMaxPool, CvtReshape, CvtDropout
from src.converter.builtin import CvtSoftmax, CvtGemm

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

        # Indicator if after conversion, 'tOp.builtinOptions' was set
        implicitOperatorType = True

        # Identify ONNX operator and convert it
        match(oNode.opType):
            case "Conv":
                tOp.builtinOptions = CvtConv.convert(oNode.attributes)
            case "Gemm":
                tOp.builtinOptions = CvtGemm.convert(oNode.attributes,
                                                     tOp,
                                                     self.__builder)
            case "LRN":
                tOp.builtinOptions = CvtLRN.convert(oNode.attributes)
            case "MaxPool":
                tOp.builtinOptions = CvtMaxPool.convert(oNode.attributes)
            case "Relu":
                tOp.builtinOptions = None
                tOp.opcodeIndex = self.__builder.opCodeIndexForOpType(tflBO.BuiltinOperator.RELU)
                implicitOperatorType = False
            case "Reshape":
                tOp.builtinOptions = CvtReshape.convert(tOp)
            case "Softmax":
                tOp.builtinOptions = CvtSoftmax.convert(oNode.attributes)


                """ Operators that might not get converted! """
            case "Dropout":
                tOp.builtinOptions = CvtDropout.convert(oNode.attributes)
                if tOp.builtinOptions is None:
                    self.__builder.skipOperator(tOp)
                return


            case _:
                implicitOperatorType = False
                err.error(None, f"Conversion of ONNX Operator '{oNode.opType}'",
                          "is not yet supported!")
                
        # Assign 'tOp' its operator type. If possible
        if implicitOperatorType:
            tOp.opcodeIndex = self.__builder.opCodeIndexForOpType(tOp.builtinOptions.operatorType)
        

        self.__builder.checkAndAppendOperator(tOp)