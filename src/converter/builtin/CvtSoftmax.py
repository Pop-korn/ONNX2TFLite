from typing import List

import src.err as err

import src.parser.builtin.Softmax as onnxSoftmax

from src.generator.builtin import(
    Softmax as tflSoftmax,
    Reshape as tflReshape
) 
import src.generator.model.Operators as tflO
import src.generator.model.Tensors as tflT

import src.converter.builder.ModelBuilder as ModelBuilder
import src.converter.conversion.Translator as Translator

import lib.tflite.BuiltinOperator as tflBOp


def getShapeForReshape(axis: int, oldShape: List[int]):
    """ If ONNX Softmax uses special 'axis', return the shape the ONNX Operator
        would reshape the input into before applying Softmax. """
    
    newShape = []
    tmp = 1
    for i, dim in enumerate(oldShape):
        if i == axis:
            newShape.append(tmp)
            tmp = dim
        else:
            tmp *= dim
    newShape.append(tmp)

    return newShape


def convert(oSM: onnxSoftmax.Softmax,
            tOp: tflO.Operator,
            modelBuilder: ModelBuilder.ModelBuilder):
    """ Convert the ONNX 'Softmax' operator to TFLite.
        Function doesn't return anything. It handles adding new operators
        to the model by itself. """
    
    if oSM.axis != -1 or oSM.axis != len(tOp.tmpInputs[0].shape.vector)-1:
        err.note(f"ONNX operator 'Softmax' has attribute 'axis' = '{oSM.axis}'.",
                 "Must add 2 'Reshape' operators to implement conversion.")

        # Special case        
        if Translator.isNHWC(tOp.tmpInputs[0].shape.vector):
            err.error(err.Code.NOT_IMPLEMENTED, "Conversion of ONNX Softmax with",
                      f"axis ='{oSM.axis}' and NCHW input is not yet supported!")
            
        
        
        """ Softmax cannot be converted normally, because of the 'axis' 
            attributes value. TFLite 'Softmax' must be surrounded by 2 'Reshape'
            operators. """
        
        oldShape = tOp.tmpInputs[0].shape.vector
        newShape = getShapeForReshape(oSM.axis, oldShape)
        
        """ Tensors taking part in this conversion. """
        X = tOp.tmpInputs[0]

        Y = tOp.tmpOutputs[0]

        T1 = modelBuilder.duplicateTensor(X, "Softmax_tmp_1_")
        T1.shape = tflT.Shape([]) # TODO calculate exact new shape

        T2 = modelBuilder.duplicateTensor(Y, "Softmax_tmp_2_")
        T2.shape = tflT.Shape([]) # TODO calculate exact new shape


        """ Create the first 'Reshape' operator. """
        reshape1 = tflO.Operator(
            builtinOptions=tflReshape.Reshape(newShape),
            opcodeIndex=modelBuilder.opCodeIndexForOpType(tflBOp.BuiltinOperator.RESHAPE)
        )
        reshape1.tmpInputs = [ X ]
        reshape1.tmpOutputs = [ T1 ]


        """ Create the 'Softmax' operator. """
        softmax = tflO.Operator(
            builtinOptions = tflSoftmax.Softmax(1.0),
            opcodeIndex=modelBuilder.opCodeIndexForOpType(tflBOp.BuiltinOperator.SOFTMAX)
        )
        softmax.tmpInputs = [ T1 ]
        softmax.tmpOutputs = [ T2 ]

        
        """ Create the secod 'Reshape' operator. """
        reshape2 = tflO.Operator(
            builtinOptions=tflReshape.Reshape(oldShape),
            opcodeIndex=modelBuilder.opCodeIndexForOpType(tflBOp.BuiltinOperator.RESHAPE)
        )
        reshape2.tmpInputs = [ T2 ]
        reshape2.tmpOutputs = [ Y ]


        """ Add the operators to the model. """
        modelBuilder.checkAndAppendOperator(reshape1)
        modelBuilder.checkAndAppendOperator(softmax)
        modelBuilder.checkAndAppendOperator(reshape2)

    else:
        """ Straightforward conversion. """

        tOp.builtinOptions = tflSoftmax.Softmax(1.0)
        tOp.opcodeIndex = modelBuilder.opCodeIndexForOpType(tflBOp.BuiltinOperator.SOFTMAX)
        modelBuilder.checkAndAppendOperator(tOp)
