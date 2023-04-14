import onnx
from onnx import shape_inference, checker

file = "data/onnx/google-speech-dataset-compact.onnx"
file = "data/onnx/cnn_trad_fpool3.onnx"
file = "data/onnx/audio.onnx"

model = shape_inference.infer_shapes_path(file,"test/t.onnx")
