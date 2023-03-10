from PIL import Image

import tensorflow as tf
import onnxruntime as ort
import onnx

import numpy as np

image = Image.open("data/224x224/cat2.jpg")
image = [np.asarray(image).tolist()]
image = np.asarray(image, np.float32)
nchwImage = np.moveaxis(image,3,1)

onnxModel = onnx.load("data/onnx/bvlcalexnet-12.onnx")
onnx.checker.check_model(onnxModel)

ortSession = ort.InferenceSession("data/onnx/bvlcalexnet-12.onnx")
onnxOutput = ortSession.run(None, {ortSession.get_inputs()[0].name : nchwImage})
onnxOutput = np.asarray(onnxOutput).squeeze()
print("ONNX:")
print(f"\tMax: output[{onnxOutput.argmax()}] = {onnxOutput.max()}")
print(f"\tSum = {onnxOutput.sum()}")
print(f"\tMean = {onnxOutput.mean()}")
print(f"\tStd = {onnxOutput.std()}")


tflModel = tf.lite.Interpreter(model_path="test/alexnet.tflite")
tflModel.allocate_tensors()



inpt = tflModel.get_input_details()
outpt = tflModel.get_output_details()

tflModel.set_tensor(inpt[0]['index'], image)

tflModel.invoke()

tflOutput:np.ndarray = tflModel.get_tensor(outpt[0]['index'])
print("TFLite:")
print(f"\tMax: output[{tflOutput.argmax()}] = {tflOutput.max()}")
print(f"\tSum = {tflOutput.sum()}")
print(f"\tMean = {tflOutput.mean()}")
print(f"\tStd = {tflOutput.std()}")