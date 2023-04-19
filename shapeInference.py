import onnx
from onnx import shape_inference, checker

file = "data/onnx/google-speech-dataset-compact.onnx"
file = "data/onnx/cnn_trad_fpool3.onnx"
file = "data/onnx/speech_command_classifier_trained.onnx"

model = shape_inference.infer_shapes_path(file,file)
