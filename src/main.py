import flatbuffers as fb

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

    subGraph = sg.SubGraph(sg.Inputs([16]), sg.Outputs([0]))

        # Operators
    operators = o.Operators()
    operators.append(o.Operator(o.Inputs([16,3,1]),o.Outputs([2]),Conv2D.Conv2D(strideH=1,strideW=1)))
    subGraph.operators = operators

        # Tensors
    tensors = t.Tensors()
    quantization = q.Quantization(q.Min([0.0]),q.Max([0.99609375]), q.Scale([0.00390625]), q.ZeroPoint([0]))
    tensors.append(t.Tensor(quantization,t.Shape([1,10]),"CifarNet/Predictions/Reshape_1"
    , 17, tt.TensorType.UINT8))
        # TODO add more
    subGraph.tensors = tensors

    subGraphs.append(subGraph)

    # Buffers
    buffers = b.Buffers([b.Buffer([1,2,3,4])])

    return m.Model(3,"TOCO Converted.",buffers,operatorCodes,subGraphs)


# Create a flatbuffer builder to build the .tflite file
builder = fb.Builder(1024)

# Build the TFLite model structure. No TFLite generated yet, only internal representation
model = BuildModel()

# Generate the TFLite fot the model
model.genTFLite(builder)

# Write the TFLite data to file
buffer = builder.Output()
with open("test/out.tflite","wb") as f:
    f.write(buffer)
