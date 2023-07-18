"""
    CvtConv

Convert ONNX operator Conv to TFLite Conv2D or Conv3D.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

from src.converter.conversion import Translator, common
import src.converter.builder.ModelBuilder as ModelBuilder

import src.err as err

import src.parser.builtin.Conv as onnxConv

from src.generator.builtin import (
    Conv2D as tflConv2D, Reshape as tflReshape, Transpose as tflTranspose
)
from src.generator.model import Operators as tflO, Tensors as tflT

import lib.tflite.BuiltinOperator as tflBOp



def __convert1DConv(oConv: onnxConv.Conv, tOp: tflO.Operator,
            modelBuilder: ModelBuilder.ModelBuilder):
    """ Handle the conversion of 1 dimensional convolution.
        TFLite doesn't support 1D convolution. It can be represented using 
        Transpose -> Reshape -> Conv2D -> Reshape -> Transpose. 
        The transpose operators only need to be generated if the 'channels' 
        dimension is not 1.
        """
        
    # TODO
    # generateTreanspose = tOp.tmpInputs[0].shape.get(1) != 1
            
    """ Calculate the shapes for equivalent 2D convolution """
    oldInputShape = tOp.tmpInputs[0].shape.vector #  NCH 
    oldOutputShape = tOp.tmpOutputs[0].shape.vector #  NCH

    nchwInputShape = Translator.nchToNchwDims(oldInputShape) # NCHW
    nchwOutputShape = Translator.nchToNchwDims(oldOutputShape) # NCHW

    nhwcInputShape = Translator.dimsToNHWC(nchwInputShape) # NHWC
    nhwcOutputShape = Translator.dimsToNHWC(nchwOutputShape) # NHWC


    """ Generate tensors taking part in the conversion """

    X = tOp.tmpInputs[0] # NCH

    T1 = modelBuilder.duplicateTensor(X, "Conv1D_1_") # NCHW
    T1.shape = tflT.Shape(nchwInputShape)

    T2 = modelBuilder.duplicateTensor(X, "Conv1D_2_") # NHWC
    T2.shape = tflT.Shape(nhwcInputShape)

    T3 = modelBuilder.duplicateTensor(X, "Conv1D_3_") # NHWC
    T3.shape = tflT.Shape(nhwcOutputShape)

    T4 = modelBuilder.duplicateTensor(X,"Conv1D_4_") # NCHW
    T4.shape = tflT.Shape(nchwOutputShape)

    Y = tOp.tmpOutputs[0] # NCH

    W = tOp.tmpInputs[1]
    W.tmpBuffer.data = Translator.nchToNhwcData(W.tmpBuffer.data, 
                                                W.shape.vector)
    W.shape = tflT.Shape(Translator.nchToNhwcDims(W.shape.vector))
    B = tOp.tmpInputs[2]

    # Transpose permutations
    P1 = modelBuilder.createTensorForData(Translator.createToNHWCPerm(nhwcInputShape),
                                          "Conv1D_perm_1_")
            
    P2 = modelBuilder.createTensorForData(Translator.createToNCHWPerm(nhwcInputShape),
                                          "Conv1D_perm_2_")



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

    # Conv
    newKernelShape = oConv.kernelShape
    newKernelShape.append(1)
    newDilations = oConv.dilations
    newDilations.append(1)
    conv = tflO.Operator(
        builtinOptions=tflConv2D.Conv2D(
            dilationHFactor=oConv.dilations[0],
            padding=Translator.convertPadding(oConv.autoPad,
                                              oConv.pads,
                                              newKernelShape,
                                              newDilations
                                             ),
            strideH=oConv.strides[0],
            strideW=1
        ),
        opcodeIndex=modelBuilder.opCodeIndexForOpType(tflBOp.BuiltinOperator.CONV_2D)
    )
    conv.tmpInputs = [ T2 , W , B ]
    conv.tmpOutputs = [ T3 ]

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
    modelBuilder.checkAndAppendOperator(conv)
    modelBuilder.checkAndAppendOperator(transpose2)
    modelBuilder.checkAndAppendOperator(reshape2)




def __convert2DConv(oConv: onnxConv.Conv, tOp: tflO.Operator,
            modelBuilder: ModelBuilder.ModelBuilder):
    """ Handle the conversion of ONNX 'Conv' operator to TFLite 'Conv2D'. """
    tConv = tflConv2D.Conv2D()

    common.assign2DStrides(tConv, oConv.strides)
    
    if common.isOfSize(oConv.dilations, 2):
        tConv.dilationHFactor = oConv.dilations[0]
        tConv.dilationWFactor = oConv.dilations[1]

    tConv.padding = Translator.convertPadding(oConv.autoPad, oConv.pads, 
                                                  oConv.kernelShape,
                                                  oConv.dilations)
        
    if len(tOp.tmpInputs) == 2:
        # Operator is has no bias. ONNX model can ommit it. TFLite can't.
        kernelShape = tOp.tmpInputs[1].shape.vector
        bias = modelBuilder.createZerosTensor([kernelShape[0]], 
                                                  "zero_conv_bias",
                                                  tOp.tmpInputs[1].tmpBuffer.data.dtype,
                                                  True)
        tOp.tmpInputs.append(bias)
        
    # Insert the new operator to the model
    tOp.builtinOptions = tConv
    tOp.opcodeIndex=modelBuilder.opCodeIndexForOpType(tflBOp.BuiltinOperator.CONV_2D)
    modelBuilder.checkAndAppendOperator(tOp)




def __convert3DConv(oConv: onnxConv.Conv, tOp: tflO.Operator,
            modelBuilder: ModelBuilder.ModelBuilder):
    err.error(err.Code.NOT_IMPLEMENTED,
                      "Conv3D NEEDS to be implemented and converted!")




def convert(oConv: onnxConv.Conv, tOp: tflO.Operator,
            modelBuilder: ModelBuilder.ModelBuilder):
    """ Convert the ONNX 'Conv' operator to TFLite. """

    if len(oConv.kernelShape) == 1:
        # 1D Convolution
        __convert1DConv(oConv, tOp, modelBuilder)

    elif len(oConv.kernelShape) == 2:
        # 2D Convolution
        __convert2DConv(oConv, tOp, modelBuilder)

    elif len(oConv.kernelShape) == 3:
        # 3D Convolution
        __convert3DConv(oConv, tOp, modelBuilder)
    else:
        err.error(err.Code.UNSUPPORTED_OPERATOR_ATTRIBUTES,
                  f"Convolution with kernel shape '{oConv.kernelShape}'",
                  "is not supported!")
