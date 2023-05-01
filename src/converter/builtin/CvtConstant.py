"""
    CvtConstant.

Convert ONNX operator Constant to a TFLite static tensor

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

import src.parser.builtin.Constant as onnxConstant

import src.converter.builder.ModelBuilder as ModelBuilder

def convert(oC: onnxConstant.Constant, 
            modelBuilder: ModelBuilder.ModelBuilder) -> None:
    
    modelBuilder.createTensorForData(oC.value.data, oC.value.name)

    return None

