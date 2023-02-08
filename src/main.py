import flatbuffers as fb

import numpy as np

import tflite.BuiltinOperator as bo
import tflite.TensorType as tt
import tflite.Padding as p

import generator.model.Model as m
import generator.model.OperatorCodes as oc
import generator.model.SubGraphs as sg
import generator.model.Tensors as t
import generator.model.Quantization as q
import generator.model.Operators as o
import generator.model.Buffers as b

import generator.builtin.Conv2D as Conv2D
import generator.builtin.MaxPool2D as MaxPool2D
import generator.builtin.FullyConnected as FullyConnected

def BuildOperators(operators: o.Operators):
    operators.append(o.Operator(o.Inputs([0,1,2]),o.Outputs([3]),
                    Conv2D.Conv2D(strideH=1,strideW=1)))

    operators.append(o.Operator(o.Inputs([3]),o.Outputs([4]),
                    MaxPool2D.MaxPool2D(p.Padding.VALID,2,2,2,2), 2))

    operators.append(o.Operator(o.Inputs([4,5,6]), o.Outputs([7]),
                    Conv2D.Conv2D(strideW=1, strideH=1)))

    operators.append(o.Operator(o.Inputs([7]),o.Outputs([8]),
                    MaxPool2D.MaxPool2D(p.Padding.VALID,2,2,2,2), 2))

    operators.append(o.Operator(o.Inputs([8,9,10]), o.Outputs([11]),
                    Conv2D.Conv2D(strideW=1, strideH=1)))

    operators.append(o.Operator(o.Inputs([11]),o.Outputs([12]),
                    MaxPool2D.MaxPool2D(p.Padding.VALID,2,2), 2))

    operators.append(o.Operator(o.Inputs([12,13,14]), o.Outputs([15]),
                    FullyConnected.FullyConnected(),1))

def BuildTensors(tensors: t.Tensors):
    # Input
    inputQ = q.Quantization(q.Min([-1.0078740119934082]),q.Max([1.0]),q.Scale([0.007874015718698502]),q.ZeroPoint([128]))
    tensors.append(t.Tensor(t.Shape([1,32,32,3]),"input", 
                    0, tt.TensorType.UINT8, inputQ))

    # Conv1
    conv1WQ = q.Quantization(q.Min([-1.6849952936172485]),q.Max([1.2710195779800415])
    ,q.Scale([0.01163785345852375]),q.ZeroPoint([146]))
    tensors.append(t.Tensor(t.Shape([32,5,5,3]),"CifarNet/conv1/weights_quant/FakeQuantWithMinMaxVars",
                    1, tt.TensorType.UINT8, conv1WQ))

    conv1BQ = q.Quantization(scale=q.Scale([0.00009163664071820676]))
    tensors.append(t.Tensor(t.Shape([32]), "CifarNet/conv1/Conv2D_bias", 
                    2, tt.TensorType.INT32, conv1BQ))

    conv1OutQ = q.Quantization(q.Min([0.0]),q.Max([23.805988311767578]), q.Scale([0.09335681796073914]))
    tensors.append(t.Tensor(t.Shape([1,32,32,32]), "CifarNet/conv1/Relu", 
                    3, tt.TensorType.UINT8, conv1OutQ))

    # MaxPool1
    maxPool1OutQ = q.Quantization(q.Min([0.0]), q.Max([23.805988311767578]), q.Scale([0.09335681796073914]))
    tensors.append(t.Tensor(t.Shape([1,16,16,32]), "CifarNet/pool1/MaxPool", 
                    4, tt.TensorType.UINT8, maxPool1OutQ))

    # Conv2
    conv2WQ = q.Quantization(q.Min([-0.8235113024711609]),q.Max([0.7808409929275513])
    ,q.Scale([0.006316347513347864]),q.ZeroPoint([131]))
    tensors.append(t.Tensor(t.Shape([32,5,5,32]),"CifarNet/conv2/weights_quant/FakeQuantWithMinMaxVars", 
                    5, tt.TensorType.UINT8, conv2WQ))

    conv2BQ = q.Quantization(scale=q.Scale([0.0005896741058677435]))
    tensors.append(t.Tensor(t.Shape([32]), "CifarNet/conv2/Conv2D_bias", 
                    6, tt.TensorType.INT32, conv2BQ))

    conv2OutQ = q.Quantization(q.Min([0.0]),q.Max([21.17963981628418]), q.Scale([0.08305741101503372]))
    tensors.append(t.Tensor(t.Shape([1,16,16,32]), "CifarNet/conv2/Relu", 
                    7, tt.TensorType.UINT8, conv2OutQ))

    # MaxPool2
    maxPool1OutQ = q.Quantization(q.Min([0.0]), q.Max([21.17963981628418]), q.Scale([0.08305741101503372]))
    tensors.append(t.Tensor(t.Shape([1,8,8,32]), "CifarNet/pool2/MaxPool", 
                    8, tt.TensorType.UINT8, maxPool1OutQ))

    # Conv3
    conv3WQ = q.Quantization(q.Min([-0.490180641412735]),q.Max([0.4940822720527649]),
                            q.Scale([0.003875050926581025]),q.ZeroPoint([127]))
    tensors.append(t.Tensor(t.Shape([64,5,5,32]),"CifarNet/conv3/weights_quant/FakeQuantWithMinMaxVars", 
                    9, tt.TensorType.UINT8, conv3WQ))

    conv3BQ = q.Quantization(scale=q.Scale([0.0003218516940250993]))
    tensors.append(t.Tensor(t.Shape([64]), "CifarNet/conv3/Conv2D_bias", 
                    10, tt.TensorType.INT32, conv3BQ))

    conv3OutQ = q.Quantization(q.Min([0.0]),q.Max([26.186586380004883]), q.Scale([0.10269249230623245]))
    tensors.append(t.Tensor(t.Shape([1,8,8,64]), "CifarNet/conv3/Relu", 
                    11, tt.TensorType.UINT8, conv3OutQ))

    # MaxPool3
    maxPool1OutQ = q.Quantization(q.Min([0.0]), q.Max([26.186586380004883]), q.Scale([0.10269249230623245]))
    tensors.append(t.Tensor(t.Shape([1,4,4,64]), "CifarNet/pool3/MaxPool", 
                    12, tt.TensorType.UINT8, maxPool1OutQ))

    # FullyConnected
    fcWQ = q.Quantization(q.Min([-0.25385990738868713]), q.Max([0.38874608278274536]), 
                            q.Scale([0.002529944758862257]), q.ZeroPoint([101]))
    tensors.append(t.Tensor(t.Shape([10,1024]), "CifarNet/logits/weights_quant/FakeQuantWithMinMaxVars/transpose",
                    13, tt.TensorType.UINT8, fcWQ))

    fcBQ = q.Quantization(scale=q.Scale([0.00025980634381994605]))
    tensors.append(t.Tensor(t.Shape([10]),"CifarNet/logits/MatMul_bias",
                    14,tt.TensorType.INT32,fcBQ))



    # Output
    outputQ = q.Quantization(q.Min([0.0]),q.Max([0.99609375]), q.Scale([0.00390625]))
    tensors.append(t.Tensor(t.Shape([1,10]), "CifarNet/Predictions/Reshape_1", 
                    17, tt.TensorType.UINT8, outputQ))

def BuildBuffers(buffers: b.Buffers):
    # Input
    buffers.append(b.Buffer())

    # Conv1
    convW = np.load("data/buffers/conv1-weights").flatten().tolist()
    buffers.append(b.Buffer(convW,tt.TensorType.UINT8))

    convB = np.load("data/buffers/conv1-bias").flatten().tolist()
    buffers.append(b.Buffer(convB))

    buffers.append(b.Buffer())

    #MaxPool1
    buffers.append(b.Buffer())

    # Conv2
    convW = np.load("data/buffers/conv2-weights").flatten().tolist()
    buffers.append(b.Buffer(convW,tt.TensorType.UINT8))

    convB = np.load("data/buffers/conv2-bias").flatten().tolist()
    buffers.append(b.Buffer(convB))

    buffers.append(b.Buffer())

    #MaxPool2
    buffers.append(b.Buffer())

    # Conv3
    convW = np.load("data/buffers/conv3-weights").flatten().tolist()
    buffers.append(b.Buffer(convW,tt.TensorType.UINT8))

    convB = np.load("data/buffers/conv3-bias").flatten().tolist()
    buffers.append(b.Buffer(convB))

    buffers.append(b.Buffer())

    #MaxPool3
    buffers.append(b.Buffer())

    # Output
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

    subGraph = sg.SubGraph(sg.Inputs([0]), sg.Outputs([17]))

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
model.Finish(builder, tflModel)

# Write the TFLite data to file
buffer = builder.Output()
with open("test/out.tflite","wb") as f:
    f.write(buffer)
