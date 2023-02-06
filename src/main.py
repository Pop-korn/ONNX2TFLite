import flatbuffers as fb

import numpy as np
import io 

import tflite.BuiltinOperator as bo
import tflite.TensorType as tt

import generator.model.Model as m
import generator.model.OperatorCodes as oc
import generator.model.SubGraphs as sg
import generator.model.Tensors as t
import generator.model.Quantization as q
import generator.model.Operators as o
import generator.model.Buffers as b

import generator.builtin.Conv2D as Conv2D

def BuildOperators(operators: o.Operators):
    operators.append(o.Operator(o.Inputs([0,1,2]),o.Outputs([3])
    ,Conv2D.Conv2D(strideH=1,strideW=1)))

def BuildTensors(tensors: t.Tensors):
    # quantization = q.Quantization(q.Min([0.0]),q.Max([0.99609375]), q.Scale([0.00390625]), q.ZeroPoint([0]))
    # tensors.append(t.Tensor(quantization,t.Shape([1,10]),"CifarNet/Predictions/Reshape_1"
    # , 17, tt.TensorType.UINT8))
    #     # TODO add more

    inputQuant = q.Quantization(q.Min([-1.0078740119934082]),q.Max([1.0]),q.Scale([0.007874015718698502]),q.ZeroPoint([128]))
    tensors.append(t.Tensor(inputQuant,t.Shape([1,32,32,3]),"input",0,tt.TensorType.UINT8))

    convWQuant = q.Quantization(q.Min([-1.6849952936172485]),q.Max([1.2710195779800415])
    ,q.Scale([0.01163785345852375]),q.ZeroPoint([146]))
    tensors.append(t.Tensor(convWQuant,t.Shape([32,5,5,3]),"CifarNet/conv1/weights_quant/FakeQuantWithMinMaxVars"
    ,1,tt.TensorType.UINT8))

    convBQuant = q.Quantization(scale=q.Scale([0.00009163664071820676 ]))
    tensors.append(t.Tensor(convBQuant,t.Shape([32]),"CifarNet/conv1/Conv2D_bias",2,tt.TensorType.INT32))

    outpuQuant = q.Quantization(q.Min([0.0]),q.Max([23.805988311767578]), q.Scale([0.09335681796073914]))
    tensors.append(t.Tensor(outpuQuant,t.Shape([1,32,32,32]),"CifarNet/conv1/Relu"
    , 3, tt.TensorType.UINT8))

def BuildBuffers(buffers: b.Buffers):
    buffers.append(b.Buffer()) # input -> empty

    # load weights from file
    convWeights = np.load("data/buffers/conv1-weights").flatten().tolist()
    buffers.append(b.Buffer(convWeights,tt.TensorType.UINT8))

    # load bias from file
    convBias = np.load("data/buffers/conv1-bias").flatten().tolist()
    buffers.append(b.Buffer(convBias))

    buffers.append(b.Buffer()) # output -> empty

def BuildModel():
    """ Generate the 'cifar10_model.tflite' """
    # OperatroCodes
    operatorCodes = oc.OperatorCodes()
    operatorCodes.append(oc.OperatorCode(bo.BuiltinOperator.CONV_2D))
    #operatorCodes.append(oc.OperatorCode(bo.BuiltinOperator.FULLY_CONNECTED))
    #operatorCodes.append(oc.OperatorCode(bo.BuiltinOperator.MAX_POOL_2D))
    #operatorCodes.append(oc.OperatorCode(bo.BuiltinOperator.SOFTMAX))

    # SubGraphs - Model only has 1 subgraph
    subGraphs = sg.SubGraphs()

    subGraph = sg.SubGraph(sg.Inputs([0]), sg.Outputs([3]))

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

    return m.Model(3,"",buffers,operatorCodes,subGraphs)


# Create a flatbuffer builder to build the .tflite file
builder = fb.Builder(2048)

# Build the TFLite model structure. No TFLite generated yet, only internal representation
model = BuildModel()

# Generate the TFLite fot the model
tflModel = model.genTFLite(builder)


builder.Finish(tflModel,"TFL3".encode("utf-8"))


# Write the TFLite data to file
buffer = builder.Output()
with open("test/out.tflite","wb") as f:
    f.write(buffer)
