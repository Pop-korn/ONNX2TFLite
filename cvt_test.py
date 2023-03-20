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
    print(f"\tShape = {output.shape}")
    print(f"\tSum = {output.sum()}")
    print(f"\tMean = {output.mean()}")
    print(f"\tStd = {output.std()}")

def tensorToNCHW(tensor: np.ndarray):
    if len(tensor.shape) < 4:
        return tensor
    return np.moveaxis(tensor,3,1)

def shapeToNHWC(shape):
    c = shape[1]
    shape[1:-2] = shape[2:-1]
    shape[-1] = c
    return shape

def loadImage(file):
    img = Image.open(file)
    img = [np.asarray(img).tolist()]
    return np.asarray(img, np.float32)

def runOnnxModel(modelFile, inpt):
    model = onnx.load(modelFile)
    onnx.checker.check_model(model)

    sess = ort.InferenceSession(modelFile)
    
    res = sess.run(None, {sess.get_inputs()[0].name : tensorToNCHW(inpt)})
    return np.asarray(res).squeeze()

def runTFLiteModel(modelFile, inpt):
    tflModel = tf.lite.Interpreter(model_path=modelFile)
    tflModel.allocate_tensors()

    inDet = tflModel.get_input_details()
    outDet = tflModel.get_output_details()

    tflModel.set_tensor(inDet[0]['index'], inpt)

    tflModel.invoke()

    return tflModel.get_tensor(outDet[0]['index'])


def keepOnlyTensors(col, tensorNames):
    toKeep = []
    for t in col:
        if t.name in tensorNames:
            toKeep.append(t)
    while len(col) > 0:
        col.pop()
    for t in toKeep:
        col.append(t)

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
        usedTensors.extend([i for i in node.input])
        usedTensors.extend([o for o in node.output])

    keepOnlyTensors(model.graph.initializer, usedTensors)
    keepOnlyTensors(model.graph.value_info, usedTensors)

    onnx.save(model,newModel)


def runAndTestFirstNOperators(originalOnnxFile, outOnnxFile, 
                             outTfliteFile, numOpsToPreserve,
                             imageFile):
    
    image = loadImage(imageFile)

    createReducedOnnxModelFrom(originalOnnxFile, outOnnxFile, numOpsToPreserve-1)

    convert.convertModel(outOnnxFile, outTfliteFile)

    onnxOut = runOnnxModel(outOnnxFile, image)

    tflOut = runTFLiteModel(outTfliteFile, image)

    printStats("ONNX", onnxOut)
    printStats("TFLite", tensorToNCHW(tflOut))


def pickOutOperators(onnxFile, startIdx, endIdx):
    model = onnx.load(onnxFile)

    inputs = []
    inputsVI = []

    outputs = []
    outputsVI = []

    tensorsToKeep = []

    nodesToRemove = []

    for i, node in enumerate(model.graph.node):
        if i == startIdx:
            inputs = [inpt for inpt in node.input]
            tensorsToKeep.extend(inputs)
        if i == endIdx:
            outputs = [outpt for outpt in node.output]
            tensorsToKeep.extend(outputs)
            tensorsToKeep.extend( [t for t in node.input] )
        if i > startIdx and i < endIdx:
            tensorsToKeep.extend( [t for t in node.input] )
        if i < startIdx or i > endIdx:
            nodesToRemove.append(node)

    for node in nodesToRemove:
        model.graph.node.remove(node)


    keepOnlyTensors(model.graph.initializer, tensorsToKeep)
    keepOnlyTensors(model.graph.value_info, tensorsToKeep)

    for vi in model.graph.value_info:
        if vi.name in inputs:
            inputsVI.append(vi)
        elif vi.name in outputs:
            outputsVI.append(vi)

    for vi in model.graph.input:
        if vi.name in inputs:
            inputsVI.append(vi)
    for vi in model.graph.output:
        if vi.name in outputs:
            outputsVI.append(vi)

    if len(outputsVI) == 0:
        for o in model.graph.output:
            outputsVI.append(o)
            o.type.tensor_type.shape.Clear()
            o.name = outputs[0]

    while len(model.graph.input) > 0:
        model.graph.input.pop()
    while len(model.graph.output) > 0:
        model.graph.output.pop()

    for i in inputsVI:
        model.graph.input.append(i)
    for o in outputsVI:
        model.graph.output.append(o)

    return model

def runAndTestOperators(originalOnnxFile, outOnnxFile, 
                         outTfliteFile, startIdx, endIdx):
    
    print(originalOnnxFile,outOnnxFile,outTfliteFile)
    onnxModel = pickOutOperators(originalOnnxFile,startIdx, endIdx)          

    print(f"\tTesting from '{onnxModel.graph.node[0].op_type}' to",
          f"'{onnxModel.graph.node[endIdx-startIdx].op_type}'")
    onnx.save(onnxModel, outOnnxFile)
    convert.convertModel(outOnnxFile, outTfliteFile)

    shape = [dim.dim_value if dim.dim_value != 0 else 1 for dim in onnxModel.graph.input[0].type.tensor_type.shape.dim]
    
    shape = shapeToNHWC(shape)

    inpt: np.ndarray = np.random.rand(*shape).astype(np.float32)

    onnxOut = runOnnxModel(outOnnxFile, inpt)
    printStats("ONNX", onnxOut)

    tflOut = runTFLiteModel(outTfliteFile, inpt)
    printStats("TFLite", tensorToNCHW(tflOut))



""" -------------------- Start of execution -------------------- """

imageFile = "data/224x224/cat2.jpg"
onnxFile = "data/onnx/tinyyolov2-8.onnx"
onnxReducedFile = "test/tinyyolo.onnx"
tflReducedFile = "test/tinyyolo.tflite"

# TESTE ALEXNET CONVERSION
# runAndTestFirstNOperators("data/onnx/bvlcalexnet-12.onnx","test/alexnet.onnx",
#                           "test/alexnet.tflite",24,imageFile)
# exit()


# shape = [1,416,416,3]
# inpt: np.ndarray = np.random.rand(*shape).astype(np.float32)
# printStats("ONNX FULL:",runOnnxModel( onnxFile ,inpt))

runAndTestOperators(onnxFile, onnxReducedFile, tflReducedFile,0,3)
exit()


from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

X = helper.make_tensor_value_info("X", TensorProto.FLOAT, ["None",3,2,2])
bias = helper.make_tensor("bias", TensorProto.FLOAT, [1,3,1,1], np.ones([3],np.float32))

# Create one output (ValueInfoProto)
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["None",3,2,2])

# Create a node (NodeProto) - This is based on Pad-11
node_def = helper.make_node(
    "Add",                  # name
    ["bias","X"],          # inputs
    ["Y"],                  # outputs
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],        # nodes
    "test-model",      # name
    [X],                # inputs
    [Y],               # outputs
    [bias]
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name="onnx-example")

print(f"The model is:\n{model_def}")
onnx.checker.check_model(model_def)
print("The model is checked!")

onnx.save(model_def,"test/test.onnx")
sess = ort.InferenceSession("test/test.onnx")
    
res = sess.run(None, {sess.get_inputs()[0].name : np.ones([2,3,2,2],np.float32)})
res = np.asarray(res).squeeze()

print(res)


runAndTestOperators("test/test.onnx", "test/test.onnx", "test/test.tflite",0,0)

