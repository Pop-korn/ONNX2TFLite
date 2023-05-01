"""
    Cvt_MatMul_Add.

Convert MatMul and Add ONNX operators to TFLite FullyConnected.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

import src.generator.model.Operators as tflO

import src.converter.builder.ModelBuilder as ModelBuilder

import numpy as np

import src.err as err

def convert(tOp: tflO.Operator, modelBuilder: ModelBuilder.ModelBuilder):
    """ Convert ONNX operators MatMul and Add to TFLite FullyConnected.
        tOp is the prepared TFLite operator, with assigned tmp inputs and 
        outputs.
        This function just checks and modifies its input tensors and adds the 
        operator to the model. """

    try:
        if tOp.tmpInputs[0].shape.len() != 3:
            # Only 3 dimensions are supported right now. Print warning and add the
            # operator as is
            err.warning("Conversion of ONNX MatMul and Add with" ,
                        f"{tOp.tmpInputs[0].shape.len()} dimenstions is not yet",
                        "supported! Resulting model may not work!")
            modelBuilder.checkAndAppendOperator(tOp)
            return
        
        """ This code was only tested on one model with this pattern.
            It may, and probably does, contain bugs! """
        
        # Operator input tensors
        X = tOp.tmpInputs[0]
        W = tOp.tmpInputs[1]

        # Change the input to NHW from NCH. This function works, but it's not
        # designed for it!
        tOp.tmpInputs[0] = modelBuilder.nchwVersionOf(X)

        # Transpose the 'weights' to work with the 'bias' shape
        W.tmpBuffer.data.shape = W.shape.vector
        W.tmpBuffer.data = np.transpose(W.tmpBuffer.data)
        W.shape.vector = W.tmpBuffer.data.shape

        # Add the operator to the model
        modelBuilder.checkAndAppendOperator(tOp)

    except Exception as e:
        print(e)
        err.error(err.Code.NOT_IMPLEMENTED, "Conversion of ONNX MatMul and Add",
                  "is limited and given model contains unsupported input shapes!")