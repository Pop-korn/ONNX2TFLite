"""
    CvtTranspose

Convert ONNX operator Transpose to TFLite Transpose.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

import src.err as err

import src.parser.builtin.Transpose as onnxTranspose

import src.generator.builtin.Transpose as tflTranspose
import src.generator.meta.meta as tflMeta
import src.generator.model.Operators as tflO

import src.converter.builder.ModelBuilder as ModelBuilder

import numpy as np

def convert(oTranspose: onnxTranspose.Transpose,
            tOp: tflO.Operator,
            modelBuilder: ModelBuilder.ModelBuilder) -> tflTranspose.Transpose:
    """ Convert ONNX 'Transpose' to TFLite 'Transpose'. """

    tTranspose = tflTranspose.Transpose()

    if oTranspose.perm is None:
        err.error(err.Code.INVALID_ONNX_OPERATOR,"ONNX Operator 'Transpose' has",
                  "missing attribute 'perm'!")
        
    # Create the 'perm' tensor
    perm = np.asarray(oTranspose.perm, np.int32)
    P = modelBuilder.createTensorForData(perm, "Transpose_perm_")

    tOp.tmpInputs.append(P)

    return tTranspose
