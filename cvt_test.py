from PIL import Image

import tensorflow as tf
import onnxruntime as ort
import onnx

import numpy as np

def printStats(formatName, output):
    print(formatName)
    print(f"\tMax: output[{output.argmax()}] = {output.max()}")
    print(f"\tSum = {output.sum()}")
    print(f"\tMean = {output.mean()}")
    print(f"\tStd = {output.std()}")

def nchw(image):
    return np.moveaxis(image,3,1)

def loadImage(file):
    img = Image.open(file)
    img = [np.asarray(img).tolist()]
    return np.asarray(img, np.float32)

def runOnnxModel(modelFile, image):
    model = onnx.load(modelFile)
    onnx.checker.check_model(model)

    sess = ort.InferenceSession(modelFile)
    
    res = sess.run(None, {sess.get_inputs()[0].name : nchw(image)})
    return np.asarray(res).squeeze()

def runTFLiteModel(modelFile, image):
    tflModel = tf.lite.Interpreter(model_path=modelFile)
    tflModel.allocate_tensors()

    inDet = tflModel.get_input_details()
    outDet = tflModel.get_output_details()

    tflModel.set_tensor(inDet[0]['index'], image)

    tflModel.invoke()

    return tflModel.get_tensor(outDet[0]['index'])

image = loadImage("data/224x224/cat2.jpg")

onnxOut = runOnnxModel("data/onnx/bvlcalexnet-12.onnx", image)

tflOut = runTFLiteModel("test/alexnet.tflite", image)

printStats("ONNX", onnxOut)
printStats("TFLite", tflOut)
