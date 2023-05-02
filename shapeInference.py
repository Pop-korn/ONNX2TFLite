"""
    shapeInference

Simple script used to infer the shapes of internal tensors of an ONNX model.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

from onnx import shape_inference

file = "data/onnx/speech_command_classifier_trained.onnx"

model = shape_inference.infer_shapes_path(file,file)
