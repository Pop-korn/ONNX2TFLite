"""
    generator_test


Generate a basic .tflite model from code. Should behave identically to the 
'/data/cifar10/cifar10_model.tflite'.
Script runs both models and prints their outputs with identical inputs.
Used to test the 'flatbuffer' library and model generation.


__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

import flatbuffers as fb

import numpy as np

import tensorflow as tf

from PIL import Image

import lib.tflite.BuiltinOperator as bo
import lib.tflite.TensorType as tt
import lib.tflite.Padding as p

import src.generator.model.Model as m
import src.generator.model.OperatorCodes as oc
import src.generator.model.SubGraphs as sg
import src.generator.model.Tensors as t
import src.generator.model.Quantization as q
import src.generator.model.Operators as o
import src.generator.model.Buffers as b

import src.generator.builtin.Conv2D as Conv2D
import src.generator.builtin.MaxPool2D as MaxPool2D
import src.generator.builtin.FullyConnected as FullyConnected
import src.generator.builtin.Softmax as Softmax

def BuildOperators(operators: o.Operators):
    operators.append(o.Operator(o.Inputs([1,2,3]),o.Outputs([4]),
                    Conv2D.Conv2D(strideH=1,strideW=1)))

    operators.append(o.Operator(o.Inputs([4]),o.Outputs([5]),
                    MaxPool2D.MaxPool2D(p.Padding.VALID,2,2,2,2), 2))

    operators.append(o.Operator(o.Inputs([5,6,7]), o.Outputs([8]),
                    Conv2D.Conv2D(strideW=1, strideH=1)))

    operators.append(o.Operator(o.Inputs([8]),o.Outputs([9]),
                    MaxPool2D.MaxPool2D(p.Padding.VALID,2,2,2,2), 2))

    operators.append(o.Operator(o.Inputs([9,10,11]), o.Outputs([12]),
                    Conv2D.Conv2D(strideW=1, strideH=1)))

    operators.append(o.Operator(o.Inputs([12]),o.Outputs([13]),
                    MaxPool2D.MaxPool2D(p.Padding.VALID,2,2), 2))

    operators.append(o.Operator(o.Inputs([13,14,15]), o.Outputs([16]),
                    FullyConnected.FullyConnected(),1))

    operators.append(o.Operator(o.Inputs([16]), o.Outputs([0]),
                    Softmax.Softmax(1.0), 3))

def BuildTensors(tensors: t.Tensors):
    # Output
    outputQ = q.Quantization(q.Min([0.0]),q.Max([0.99609375]), q.Scale([0.00390625]))
    tensors.append(t.Tensor(t.Shape([1,10]), "CifarNet/Predictions/Reshape_1", 
                    0, tt.TensorType.UINT8, outputQ))

    # Input
    inputQ = q.Quantization(q.Min([-1.0078740119934082]),q.Max([1.0]),q.Scale([0.007874015718698502]),q.ZeroPoint([128]))
    tensors.append(t.Tensor(t.Shape([1,32,32,3]),"input", 
                    1, tt.TensorType.UINT8, inputQ))

    # Conv1
    conv1WQ = q.Quantization(q.Min([-1.6849952936172485]),q.Max([1.2710195779800415])
    ,q.Scale([0.01163785345852375]),q.ZeroPoint([146]))
    tensors.append(t.Tensor(t.Shape([32,5,5,3]),"CifarNet/conv1/weights_quant/FakeQuantWithMinMaxVars",
                    2, tt.TensorType.UINT8, conv1WQ))

    conv1BQ = q.Quantization(scale=q.Scale([0.00009163664071820676]))
    tensors.append(t.Tensor(t.Shape([32]), "CifarNet/conv1/Conv2D_bias", 
                    3, tt.TensorType.INT32, conv1BQ))

    conv1OutQ = q.Quantization(q.Min([0.0]),q.Max([23.805988311767578]), q.Scale([0.09335681796073914]))
    tensors.append(t.Tensor(t.Shape([1,32,32,32]), "CifarNet/conv1/Relu", 
                    4, tt.TensorType.UINT8, conv1OutQ))

    # MaxPool1
    maxPool1OutQ = q.Quantization(q.Min([0.0]), q.Max([23.805988311767578]), q.Scale([0.09335681796073914]))
    tensors.append(t.Tensor(t.Shape([1,16,16,32]), "CifarNet/pool1/MaxPool", 
                    5, tt.TensorType.UINT8, maxPool1OutQ))

    # Conv2
    conv2WQ = q.Quantization(q.Min([-0.8235113024711609]),q.Max([0.7808409929275513])
    ,q.Scale([0.006316347513347864]),q.ZeroPoint([131]))
    tensors.append(t.Tensor(t.Shape([32,5,5,32]),"CifarNet/conv2/weights_quant/FakeQuantWithMinMaxVars", 
                    6, tt.TensorType.UINT8, conv2WQ))

    conv2BQ = q.Quantization(scale=q.Scale([0.0005896741058677435]))
    tensors.append(t.Tensor(t.Shape([32]), "CifarNet/conv2/Conv2D_bias", 
                    7, tt.TensorType.INT32, conv2BQ))

    conv2OutQ = q.Quantization(q.Min([0.0]),q.Max([21.17963981628418]), q.Scale([0.08305741101503372]))
    tensors.append(t.Tensor(t.Shape([1,16,16,32]), "CifarNet/conv2/Relu", 
                    8, tt.TensorType.UINT8, conv2OutQ))

    # MaxPool2
    maxPool1OutQ = q.Quantization(q.Min([0.0]), q.Max([21.17963981628418]), q.Scale([0.08305741101503372]))
    tensors.append(t.Tensor(t.Shape([1,8,8,32]), "CifarNet/pool2/MaxPool", 
                    9, tt.TensorType.UINT8, maxPool1OutQ))

    # Conv3
    conv3WQ = q.Quantization(q.Min([-0.490180641412735]),q.Max([0.4940822720527649]),
                            q.Scale([0.003875050926581025]),q.ZeroPoint([127]))
    tensors.append(t.Tensor(t.Shape([64,5,5,32]),"CifarNet/conv3/weights_quant/FakeQuantWithMinMaxVars", 
                    10, tt.TensorType.UINT8, conv3WQ))

    conv3BQ = q.Quantization(scale=q.Scale([0.0003218516940250993]))
    tensors.append(t.Tensor(t.Shape([64]), "CifarNet/conv3/Conv2D_bias", 
                    11, tt.TensorType.INT32, conv3BQ))

    conv3OutQ = q.Quantization(q.Min([0.0]),q.Max([26.186586380004883]), q.Scale([0.10269249230623245]))
    tensors.append(t.Tensor(t.Shape([1,8,8,64]), "CifarNet/conv3/Relu", 
                    12, tt.TensorType.UINT8, conv3OutQ))

    # MaxPool3
    maxPool1OutQ = q.Quantization(q.Min([0.0]), q.Max([26.186586380004883]), q.Scale([0.10269249230623245]))
    tensors.append(t.Tensor(t.Shape([1,4,4,64]), "CifarNet/pool3/MaxPool", 
                    13, tt.TensorType.UINT8, maxPool1OutQ))

    # FullyConnected
    fcWQ = q.Quantization(q.Min([-0.25385990738868713]), q.Max([0.38874608278274536]), 
                            q.Scale([0.002529944758862257]), q.ZeroPoint([101]))
    tensors.append(t.Tensor(t.Shape([10,1024]), "CifarNet/logits/weights_quant/FakeQuantWithMinMaxVars/transpose",
                    14, tt.TensorType.UINT8, fcWQ))

    fcBQ = q.Quantization(scale=q.Scale([0.00025980634381994605]))
    tensors.append(t.Tensor(t.Shape([10]),"CifarNet/logits/MatMul_bias",
                    15,tt.TensorType.INT32,fcBQ))

    fcOutQ = q.Quantization(q.Min([-13.407316207885742]),q.Max([22.58074378967285]),
                    q.Scale([0.14112964272499084]),q.ZeroPoint([95]))
    tensors.append(t.Tensor(t.Shape([1,10]),"CifarNet/logits/BiasAdd",
                    16, tt.TensorType.UINT8,fcOutQ))



def BuildBuffers(buffers: b.Buffers):
    # Output
    buffers.append(b.Buffer())

    # Input
    buffers.append(b.Buffer())

    # Conv1
    convW = np.load("data/buffers/conv1-weights").flatten()
    buffers.append(b.Buffer(convW,tt.TensorType.UINT8))

    convB = np.load("data/buffers/conv1-bias").flatten()
    buffers.append(b.Buffer(convB))

    buffers.append(b.Buffer())

    #MaxPool1
    buffers.append(b.Buffer())

    # Conv2
    convW = np.load("data/buffers/conv2-weights").flatten()
    buffers.append(b.Buffer(convW,tt.TensorType.UINT8))

    convB = np.load("data/buffers/conv2-bias").flatten()
    buffers.append(b.Buffer(convB))

    buffers.append(b.Buffer())

    # MaxPool2
    buffers.append(b.Buffer())

    # Conv3
    convW = np.load("data/buffers/conv3-weights").flatten()
    buffers.append(b.Buffer(convW,tt.TensorType.UINT8))

    convB = np.load("data/buffers/conv3-bias").flatten()
    buffers.append(b.Buffer(convB))

    buffers.append(b.Buffer())

    # MaxPool3
    buffers.append(b.Buffer())

    # Fully Connected
    fcW = np.load("data/buffers/fc-weights").flatten()
    buffers.append(b.Buffer(fcW,tt.TensorType.UINT8))

    fcB = np.load("data/buffers/fc-bias").flatten()
    buffers.append(b.Buffer(fcB))

    buffers.append(b.Buffer())


def BuildModel():
    """ Generate the 'cifar10_model.tflite' """
    # OperatroCodes
    operatorCodes = oc.OperatorCodes()
    operatorCodes.append(oc.OperatorCode(bo.BuiltinOperator.CONV_2D))
    operatorCodes.append(oc.OperatorCode(bo.BuiltinOperator.FULLY_CONNECTED))
    operatorCodes.append(oc.OperatorCode(bo.BuiltinOperator.MAX_POOL_2D))
    operatorCodes.append(oc.OperatorCode(bo.BuiltinOperator.SOFTMAX))

    # SubGraphs - Model only has 1 subgraph
    subGraphs = sg.SubGraphs()

    subGraph = sg.SubGraph(sg.Inputs([1]), sg.Outputs([0]))

    # Operators
    operators = o.Operators()
    BuildOperators(operators)
    subGraph.operators = operators

    # Tensors
    tensors = t.Tensors()
    BuildTensors(tensors)
    subGraph.tensors = tensors

    subGraphs.append(subGraph)

    # Buffers
    buffers = b.Buffers()
    BuildBuffers(buffers)

    return m.Model(3,"TOCO Converted.",buffers,operatorCodes,subGraphs)


# Create a flatbuffer builder to build the .tflite file
builder = fb.Builder(3400)

# Build the TFLite model structure. No TFLite generated yet, only internal representation
model = BuildModel()

# Generate the TFLite fot the model
tflModel = model.genTFLite(builder)

# Write the TFLite data to file
genModelFile = "test/cifar10_model_GENERATED.tflite"
buffer = builder.Output()
with open(genModelFile,"wb") as f:
    f.write(buffer)


""" Compare the output of generated model with the pre-trained model. """


def loadModels():
    generatedModel = tf.lite.Interpreter(model_path = genModelFile)
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

runTest()
