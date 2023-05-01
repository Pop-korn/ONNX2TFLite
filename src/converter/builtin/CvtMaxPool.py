"""
    CvtMaxPool

Convert ONNX operator MaxPool to TFLite MaxPool2D.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

from src.converter.conversion import Translator, common
import src.converter.builder.ModelBuilder as ModelBuilder
import src.err as err

import lib.tflite.BuiltinOperator as tflBOp

import src.parser.builtin.MaxPool as onnxMaxPool

from src.generator.builtin import (
    MaxPool2D as tflMaxPool2D,
    Reshape as tflReshape,
    Transpose as tflTranspose
)
import src.generator.meta.meta as tflMeta
from src.generator.model import (
    Operators as tflO,
    Tensors as tflT
)


def __convert1DMaxPool(oMP: onnxMaxPool.MaxPool,
                       tOp: tflO.Operator,
                       modelBuilder: ModelBuilder.ModelBuilder):
    """ Convert the ONNX 1D 'MaxPool' operator to TFLite 'MaxPool2D'. 
        TFLite doen't support 1D MaxPool. This behaviour can be repsesented 
        using Reshape -> Transpose -> MaxPool2D -> Transpose -> Reshape. """
    
    # TODO
    # generateTreanspose = tOp.tmpInputs[0].shape.get(1) != 1
            
    """ Calculate the shapes for equivalent 2D MaxPool """
    oldInputShape = tOp.tmpInputs[0].shape.vector #  NCH 
    oldOutputShape = tOp.tmpOutputs[0].shape.vector #  NCH
            
    nchwInputShape = Translator.nchToNchwDims(oldInputShape) # NCHW
    nchwOutputShape = Translator.nchToNchwDims(oldOutputShape) # NCHW

    nhwcInputShape = Translator.dimsToNHWC(nchwInputShape) # NHWC
    nhwcOutputShape = Translator.dimsToNHWC(nchwOutputShape) # NHWC


    """ Generate tensors taking part in the conversion """

    X = tOp.tmpInputs[0] # NCH

    T1 = modelBuilder.duplicateTensor(X, "MaxPool1D_1_") # NCHW
    T1.shape = tflT.Shape(nchwInputShape)

    T2 = modelBuilder.duplicateTensor(X, "MaxPool1D_2_") # NHWC
    T2.shape = tflT.Shape(nhwcInputShape)

    T3 = modelBuilder.duplicateTensor(X, "MaxPool1D_3_") # NHWC
    T3.shape = tflT.Shape(nhwcOutputShape)

    T4 = modelBuilder.duplicateTensor(X,"MaxPool1D_4_") # NCHW
    T4.shape = tflT.Shape(nchwOutputShape)

    Y = tOp.tmpOutputs[0] # NCH

    # Transpose permutations
    P1 = modelBuilder.createTensorForData(Translator.createToNHWCPerm(nhwcInputShape),
                                          "MaxPool1D_perm_1_")
            
    P2 = modelBuilder.createTensorForData(Translator.createToNCHWPerm(nhwcInputShape),
                                          "MaxPool1D_perm_2_")



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

    # MaxPool
    newKernelShape = oMP.kernelShape
    newKernelShape.append(1)
    maxPool = tflO.Operator(
        builtinOptions=tflMaxPool2D.MaxPool2D(
            filterH=oMP.kernelShape[0],
            filterW=1,
            padding=Translator.convertPadding(oMP.autoPad,
                                              oMP.pads,
                                              newKernelShape,
                                              [1,1]
                                             ),
            strideH=oMP.strides[0],
            strideW=1
        ),
        opcodeIndex=modelBuilder.opCodeIndexForOpType(tflBOp.BuiltinOperator.MAX_POOL_2D)
    )
    maxPool.tmpInputs = [ T2 ]
    maxPool.tmpOutputs = [ T3 ]

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
    modelBuilder.checkAndAppendOperator(maxPool)
    modelBuilder.checkAndAppendOperator(transpose2)
    modelBuilder.checkAndAppendOperator(reshape2)


def __convert2DMaxPool(oMP: onnxMaxPool.MaxPool,
                       tOp: tflO.Operator,
                       modelBuilder: ModelBuilder.ModelBuilder):
    """ Convert the ONNX 2D 'MaxPool' operator to TFLite 'MaxPool2D'. """
            
    tMP = tflMaxPool2D.MaxPool2D()

    common.assign2DStrides(tMP, oMP.strides)

    if common.isOfSize(oMP.kernelShape, 2):
        tMP.filterH = oMP.kernelShape[0]
        tMP.filterW = oMP.kernelShape[1]

    tMP.padding = Translator.convertPadding(oMP.autoPad, oMP.pads, 
                                            oMP.kernelShape,
                                            oMP.dilations)


    if oMP.dilations is not None:
        err.warning("MaxPool dilations cannot be converted to TFLite!")
        
    # Add the operator to the model
    tOp.builtinOptions = tMP
    tOp.opcodeIndex=modelBuilder.opCodeIndexForOpType(tflBOp.BuiltinOperator.MAX_POOL_2D)
    modelBuilder.checkAndAppendOperator(tOp)


def convert(oMP: onnxMaxPool.MaxPool,
            tOp: tflO.Operator,           
            modelBuilder: ModelBuilder.ModelBuilder):
    match len(oMP.kernelShape):
        case 1:
            # 1D MaxPool
            __convert1DMaxPool(oMP, tOp, modelBuilder)
        case 2:
            # 2D MaxPool
            __convert2DMaxPool(oMP, tOp, modelBuilder)

        case _:
            err.error(err.Code.NOT_IMPLEMENTED,f"MaxPool with kernel shape '{oMP.kernelShape}'",
                      "is not supported!")
