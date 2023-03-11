from PIL import Image

import tensorflow as tf
import onnxruntime as ort
import onnx

import numpy as np

import src.converter.convert as convert

""" This file provides functions to  """

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

def createReducedOnnxModelFrom(fromModel, newModel, lastNode):
    model = onnx.load(fromModel)

    for vi in model.graph.value_info:
        if vi.name == model.graph.node[lastNode].output[0]:
            model.graph.output[0].name = vi.name
            model.graph.output[0].doc_string = vi.doc_string
            model.graph.output[0].type.tensor_type.elem_type = vi.type.tensor_type.elem_type
            for i in range(len(model.graph.output[0].type.tensor_type.shape.dim)):
                model.graph.output[0].type.tensor_type.shape.dim.pop()
            for i in range(len(vi.type.tensor_type.shape.dim)):
                model.graph.output[0].type.tensor_type.shape.dim.append(vi.type.tensor_type.shape.dim[i])

    while len(model.graph.node) > lastNode + 1:
        model.graph.node.pop()

    usedTensors = []
    for node in model.graph.node:
        for inpt in node.input:
            usedTensors.append(inpt)
        for outpt in node.output:
            usedTensors.append(outpt)

    tensorsToKeep = []
    for tensor in model.graph.initializer:
        if tensor.name in usedTensors:
            tensorsToKeep.append(tensor)

    while len(model.graph.initializer) != 0:
        model.graph.initializer.pop()

    for tensor in tensorsToKeep:
        model.graph.initializer.append(tensor)

    tensorsToKeep = []

    for tensor in model.graph.value_info:
        if tensor.name in usedTensors:
            tensorsToKeep.append(tensor)

    while len(model.graph.value_info) != 0:
        model.graph.value_info.pop()

    for tensor in tensorsToKeep:
        model.graph.value_info.append(tensor)

    onnx.save(model,newModel)


def reduceConvertAndTestModel(originalOnnxFile, outOnnxFile, 
                             outTfliteFile, numOpsToPreserve,
                             imageFile):
    
    image = loadImage(imageFile)

    createReducedOnnxModelFrom(originalOnnxFile, outOnnxFile, numOpsToPreserve-1)

    convert.convertModel(outOnnxFile, outTfliteFile)

    onnxOut = runOnnxModel(outOnnxFile, image)

    tflOut = runTFLiteModel(outTfliteFile, image)

    printStats("ONNX", onnxOut)
    printStats("TFLite", tflOut)




""" -------------------- Start of execution -------------------- """


imageFile = "data/224x224/cat1.jpg"
onnxFile = "data/onnx/bvlcalexnet-12.onnx"

reduceConvertAndTestModel(onnxFile, "test/alexnet_reduced.onnx",
                          "test/alexnet_reduced.tflite", 16, 
                          imageFile)