import tensorflow as tf
import numpy as np

from PIL import Image

def loadModels():
    generatedModel = tf.lite.Interpreter(model_path = "./test/out.tflite")
    generatedModel.allocate_tensors()

    premadeModel = tf.lite.Interpreter(model_path = "./data/cifar10/cifar10_model.tflite")
    premadeModel.allocate_tensors()

    return (generatedModel,premadeModel)

def loadData():
    image = Image.open("./data/cifar10/airplane1.png")
    image = [np.asarray(image).tolist()]
    image = np.asarray(image,np.uint8)

    return image


def classify(model, image):
    # Get input and output tensors.
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    model.set_tensor(input_details[0]['index'], image)

    model.invoke()

    output_data = model.get_tensor(output_details[0]['index'])
    print(output_data)


def runTest():
    generatedModel, premadeModel = loadModels()
    image = loadData()

    classify(generatedModel, image)
    classify(premadeModel, image)
