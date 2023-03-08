import numpy as np

import src.parser.builtin.Gemm as onnxGemm

import src.generator.builtin.FullyConnected as tflFullyConnected
import src.generator.meta.meta as meta
import src.generator.model.Operators as tflO

import src.converter.builder.ModelBuilder as ModelBuilder

import src.err as err

def __notYetSupported(oGemm: onnxGemm.Gemm):
    """ Print error message and exit program. """
    err.error(err.Code.UNSUPPORTED_OPERATOR_ATTRIBUTES, "Conversion of",
              f" ONNX operatro 'GEMM' with attributes: alpha='{oGemm.alpha}'",
              f"beta='{oGemm.beta}' trasnA='{oGemm.transA}'",
              f"trasnB='{oGemm.transB}' is not yet supported!")
    
    
def convert(oGemm: onnxGemm.Gemm, 
            tflOperator: tflO.Operator,
            modelBuilder: ModelBuilder.ModelBuilder) -> meta.BuiltinOptions:
    
    if oGemm.alpha == 1.0 and oGemm.beta == 1.0:
        """ No tensor multiplication by scalars required """
        
        tensorA = tflOperator.tmpInputs[0]
        tensorB = tflOperator.tmpInputs[1]
        
        if oGemm.transA:
            __notYetSupported(oGemm)

        if oGemm.transB:
            """ Input tensorB needs to be transposed. """
            
            if modelBuilder.tensorHasData(tensorB):
                """ A stored tensor needs to be transposed. This can be done
                    statically. Create a new tensor with transposed data and
                    assign it as the operators input instead. """
                
                transposedTensor = modelBuilder.createTransposedTensor(tensorB)
                tflOperator.tmpInputs[1] = transposedTensor
                
            else:
                """ Tensor has no data. It is the graph input or the output
                    of a previous operator. Need to generate 'transpose'
                    operator. """
                __notYetSupported(oGemm)

        """ Inputs have been transposed. Now convert the operator. """
        print("ahoj")
                

    else:
        __notYetSupported(oGemm)
        
