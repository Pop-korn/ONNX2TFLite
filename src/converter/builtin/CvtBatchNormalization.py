import numpy as np

import lib.tflite.BuiltinOperator as tflBOp

import src.parser.builtin.BatchNormalization as onnxBN

import src.err as err

from src.generator.builtin import(
    Add as tflAdd,
    Sub as tflSub,
    Mul as tflMul
)

from src.generator.model import(
    Operators as tflO,
    Tensors as tflT
)

import src.converter.builder.ModelBuilder as ModelBuilder


def isZeroTensor(t: tflT.Tensor) -> bool:
    return all([val == 0 for val in t.tmpBuffer.data])


def convert(oBN: onnxBN.BatchNormalization, 
            tOp: tflO.Operator,
            modelBuilder: ModelBuilder.ModelBuilder):
    
    X = tOp.tmpInputs[0]
    scale = tOp.tmpInputs[1]
    bias = tOp.tmpInputs[2]
    mean = tOp.tmpInputs[3]
    var = tOp.tmpInputs[4]

    """ TFLite does not support a BatchNormalization operator, we have to 
        simulate it with Mul and Add. 
        ONNX BatchNormalization implements the following equation:

            Y = scale * (X - mean) / sqrt(var + eps) + bias

        It can be rewritten as:

            Y = X * ( scale / ( sqrt(var + eps) ) ) + 
            + ( bias - mean * scale / ( sqrt(var + eps) ) )

        where ( scale / ( sqrt(var + eps) ) ) and 
        ( bias - mean * scale / ( sqrt(var + eps) ) ) are static tensors.
    """

        
    # Calculate the static portion of the expression
    tmp = scale.tmpBuffer.data / np.sqrt(var.tmpBuffer.data + oBN.epsilon)
    invDenom = modelBuilder.createTensorForData(tmp, "BatchNorm_invdenom")
    
    if not isZeroTensor(mean):
        """ Mean is not just all zeros. Need to claculate the second static 
            tensor. Use it as the new value for 'bias'. """
        tmp = mean.tmpBuffer.data * tmp
        bias.tmpBuffer.data -= tmp

    # Create 'Mul' operator, to multiply the input with the static tensor
    mul = tflO.Operator(
        builtinOptions= tflMul.Mul(),
        opcodeIndex=modelBuilder.opCodeIndexForOpType(tflBOp.BuiltinOperator.MUL)
    )
    mul.tmpInputs = [X,invDenom]
    fraction = modelBuilder.duplicateTensor(X,"BatchNorm_frac")
    mul.tmpOutputs = [fraction]

    # Create 'Add' operator to add 'bias' to the previous result
    add = tflO.Operator(
        builtinOptions=tflAdd.Add(),
        opcodeIndex=modelBuilder.opCodeIndexForOpType(tflBOp.BuiltinOperator.ADD)
    )
    add.tmpInputs = [fraction, bias]
    add.tmpOutputs = tOp.tmpOutputs

    # Add the operators
    modelBuilder.checkAndAppendOperator(mul)
    modelBuilder.checkAndAppendOperator(add)

