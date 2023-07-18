"""
    CvtAveragePool.

Convert ONNX operator AveragePool to TFLite AveragePool2D.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

from src.converter.conversion import Translator, common
import src.converter.builder.ModelBuilder as ModelBuilder
import src.err as err

import lib.tflite.BuiltinOperator as tflBOp

import src.parser.builtin.AveragePool as onnxAveragePool

from src.generator.builtin import (
    AveragePool2D as tflAveragePool2D,
    Reshape as tflReshape,
    Transpose as tflTranspose
)
import src.generator.meta.meta as tflMeta
from src.generator.model import (
    Operators as tflO,
    Tensors as tflT
)


def __convert1DAveragePool(oAP: onnxAveragePool.AveragePool,
                       tOp: tflO.Operator,
                       modelBuilder: ModelBuilder.ModelBuilder):
    """ Convert the ONNX 1D 'AveragePool' operator to TFLite 'AveragePool2D'. 
        TFLite doen't support 1D AveragePool. This behaviour can be repsesented 
        using Reshape -> Transpose -> AveragePool2D -> Transpose -> Reshape. """
    
    # TODO
    # generateTreanspose = tOp.tmpInputs[0].shape.get(1) != 1
            
    """ Calculate the shapes for equivalent 2D AveragePool """
    oldInputShape = tOp.tmpInputs[0].shape.vector #  NCH 
    oldOutputShape = tOp.tmpOutputs[0].shape.vector #  NCH
            
    nchwInputShape = Translator.nchToNchwDims(oldInputShape) # NCHW
    nchwOutputShape = Translator.nchToNchwDims(oldOutputShape) # NCHW

    nhwcInputShape = Translator.dimsToNHWC(nchwInputShape) # NHWC
    nhwcOutputShape = Translator.dimsToNHWC(nchwOutputShape) # NHWC


    """ Generate tensors taking part in the conversion """

    X = tOp.tmpInputs[0] # NCH

    T1 = modelBuilder.duplicateTensor(X, "AveragPool1D_1_") # NCHW
    T1.shape = tflT.Shape(nchwInputShape)

    T2 = modelBuilder.duplicateTensor(X, "AveragPool1D_2_") # NHWC
    T2.shape = tflT.Shape(nhwcInputShape)

    T3 = modelBuilder.duplicateTensor(X, "AveragPool1D_3_") # NHWC
    T3.shape = tflT.Shape(nhwcOutputShape)

    T4 = modelBuilder.duplicateTensor(X,"AveragPool1D_4_") # NCHW
    T4.shape = tflT.Shape(nchwOutputShape)

    Y = tOp.tmpOutputs[0] # NCH

    # Transpose permutations
    P1 = modelBuilder.createTensorForData(Translator.createToNHWCPerm(nhwcInputShape),
                                          "AveragPool1D_perm_1_")
            
    P2 = modelBuilder.createTensorForData(Translator.createToNCHWPerm(nhwcInputShape),
                                          "AveragPool1D_perm_2_")



    """ Create the new operators """
    # Reshape 1
    reshape1 = tflO.Operator(
        builtinOptions=tflReshape.Reshape(nchwInputShape),
        opcodeIndex=modelBuilder.opCodeIndexForOpType(tflBOp.BuiltinOperator.RESHAPE)
    )
    reshape1.tmpInputs = [ X ]
    reshape1.tmpOutputs = [ T1 ]

    # Transpose 1
    transpose1 = tflO.Operator(
        builtinOptions=tflTranspose.Transpose(),
        opcodeIndex=modelBuilder.opCodeIndexForOpType(tflBOp.BuiltinOperator.TRANSPOSE)
    )
    transpose1.tmpInputs = [ T1 , P1 ]
    transpose1.tmpOutputs = [ T2 ]

    # AveragePool
    newKernelShape = oAP.kernelShape
    newKernelShape.append(1)
    averagePool = tflO.Operator(
        builtinOptions=tflAveragePool2D.AveragePool2D(
            filterH=oAP.kernelShape[0],
            filterW=1,
            padding=Translator.convertPadding(oAP.autoPad,
                                              oAP.pads,
                                              newKernelShape,
                                              [1,1]
                                             ),
            strideH=oAP.strides[0],
            strideW=1
        ),
        opcodeIndex=modelBuilder.opCodeIndexForOpType(tflBOp.BuiltinOperator.AVERAGE_POOL_2D)
    )
    averagePool.tmpInputs = [ T2 ]
    averagePool.tmpOutputs = [ T3 ]

    # Transpose 2
    transpose2 = tflO.Operator(
        builtinOptions=tflTranspose.Transpose(),
        opcodeIndex=modelBuilder.opCodeIndexForOpType(tflBOp.BuiltinOperator.TRANSPOSE)
    )
    transpose2.tmpInputs = [ T3 , P2 ]
    transpose2.tmpOutputs = [ T4 ]

    # Reshape 2
    reshape2 = tflO.Operator(
        builtinOptions=tflReshape.Reshape(oldOutputShape),
        opcodeIndex=modelBuilder.opCodeIndexForOpType(tflBOp.BuiltinOperator.RESHAPE)
    )
    reshape2.tmpInputs = [ T4 ]
    reshape2.tmpOutputs = [ Y ]


    """ Add the new operators to the model """
    modelBuilder.checkAndAppendOperator(reshape1)
    modelBuilder.checkAndAppendOperator(transpose1)
    modelBuilder.checkAndAppendOperator(averagePool)
    modelBuilder.checkAndAppendOperator(transpose2)
    modelBuilder.checkAndAppendOperator(reshape2)


def __convert2DAveragePool(oAP: onnxAveragePool.AveragePool,
                       tOp: tflO.Operator,
                       modelBuilder: ModelBuilder.ModelBuilder):
    """ Convert the ONNX 2D 'AveragePool' operator to TFLite 'AveragePool2D'. """
            
    tAP = tflAveragePool2D.AveragePool2D()

    common.assign2DStrides(tAP, oAP.strides)

    if common.isOfSize(oAP.kernelShape, 2):
        tAP.filterH = oAP.kernelShape[0]
        tAP.filterW = oAP.kernelShape[1]

    tAP.padding = Translator.convertPadding(oAP.autoPad, oAP.pads, 
                                            oAP.kernelShape,
                                            oAP.dilations)


    if oAP.dilations is not None:
        err.warning("AveragePool dilations cannot be converted to TFLite!")
        
    # Add the operator to the model
    tOp.builtinOptions = tAP
    tOp.opcodeIndex=modelBuilder.opCodeIndexForOpType(tflBOp.BuiltinOperator.AVERAGE_POOL_2D)
    modelBuilder.checkAndAppendOperator(tOp)


def convert(oAP: onnxAveragePool.AveragePool,
            tOp: tflO.Operator,           
            modelBuilder: ModelBuilder.ModelBuilder):
    """ Convert the ONNX AveragePool operator to TFLite.  """
    
    if len(oAP.kernelShape) == 1:
        # 1D AveragePool
        __convert1DAveragePool(oAP, tOp, modelBuilder)
    elif len(oAP.kernelShape) == 2:
        # 2D AveragePool
        __convert2DAveragePool(oAP, tOp, modelBuilder)

    else:
        err.error(err.Code.NOT_IMPLEMENTED,"AveragePool with kernel shape",
                  f"'{oAP.kernelShape}' is not supported!")
