from PIL import Image

import tensorflow as tf
import onnxruntime as ort
import onnx

import numpy as np

import src.converter.convert as convert

import src.err as err

# Enable all output messages of the converter
err.MIN_OUTPUT_IMPORTANCE = err.MessageImportance.LOWEST


def printStats(formatName, output):
    """ Print some statistical information about the given numpy arrray. """

    # Find the two highest values
    secMax = 0
    max = 0
    for el in output.flatten():
        if el >= max:
            secMax = max
            max = el

    print(formatName)
    print(f"\tMax: (index {output.argmax()}) = {output.max()}")
    print(f"\tSecond Max = {secMax}")
    print(f"\tShape = {output.shape}")
    print(f"\tSum = {output.sum()}")
    print(f"\tMean = {output.mean()}")
    print(f"\tStd = {output.std()}")

def tensorToNCHW(tensor: np.ndarray):
    """ Determine if given static 'tensor' is in the NCHW format, if so, convert it.
        """
    if len(tensor.shape) < 4:
        return tensor
    return np.moveaxis(tensor, len(tensor.shape) - 1, 1)

def shapeToNHWC(shape):
    """ Convert NCHW shape to NHWC and return it. If 'shape' is not in NCHW, 
        nothing is done to it. """
    if len(shape) < 4:
        return shape
    
    shape.append( shape.pop(1) ) # Move 'channels' (idx 1) to the end

    return shape

def loadImage(file):
    """ Load image from 'file' and return it. """
    img = Image.open(file)
    img = [np.asarray(img).tolist()]
    return np.asarray(img, np.float32)

def runOnnxModel(modelFile, inpt):
    """ Run ONNX model stored in 'modelFile' with 'inpt' as input. 
        Return the output tensor. """
    model = onnx.load(modelFile)
    onnx.checker.check_model(model)

    sess = ort.InferenceSession(modelFile)
    
    res = sess.run(None, {sess.get_inputs()[0].name : tensorToNCHW(inpt)})
    return np.asarray(res).squeeze()

def runTFLiteModel(modelFile, inpt):
    """ Run TFLite model stored in 'modelFile' with 'inpt' as input. 
        Return the output tensor. """
    tflModel = tf.lite.Interpreter(model_path=modelFile)
    tflModel.allocate_tensors()

    inDet = tflModel.get_input_details()
    outDet = tflModel.get_output_details()

    tflModel.set_tensor(inDet[0]['index'], inpt)

    tflModel.invoke()

    return tflModel.get_tensor(outDet[0]['index'])


def keepOnlyTensors(col, tensorNames):
    """ Remove all tensors from collection 'col' that don't have names in
        'tensorNames'. 'col' is expected to be a ONNX 'repeated' attribute and
        items from it cannot be removed easily. """
    toKeep = []
    for t in col:
        if t.name in tensorNames:
            toKeep.append(t)
    while len(col) > 0:
        col.pop()
    for t in toKeep:
        col.append(t)

def createReducedOnnxModelFrom(fromModel, newModel, lastNode):
    """ Load ONNX model from 'fromModel', reduce is so it only contains nodes 
        upto 'lastNode'. Stroe the reduced model in 'newModel' file. """
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
    """ Convert the ONNX model in 'originalOnnxFile'. Reduce it to only contain
        the first 'numOpsToPreserve' operators and save the model in 
        'outOnnxFile'. Convert the reduced model to TFLite and save it in
        'outTfliteFile'. Then run both models with the 'imageFile' as input
        and print output statistics. """
    
    image = loadImage(imageFile)

    createReducedOnnxModelFrom(originalOnnxFile, outOnnxFile, numOpsToPreserve-1)

    convert.convertModel(outOnnxFile, outTfliteFile)

    onnxOut = runOnnxModel(outOnnxFile, image)

    tflOut = tensorToNCHW(runTFLiteModel(outTfliteFile, image))

    printStats("ONNX", onnxOut)
    printStats("TFLite", tflOut)
    printStats("Difference", onnxOut-tflOut)


def pickOutOperators(onnxFile, startIdx, endIdx):
    """ Load ONNX model in 'onnxFile'. Keep only its operators with indices from
        'starIdx' to 'endIdx' and only the used tensors. 
        Return the reduced model. """
    
    model = onnx.load(onnxFile)

    inputs = []
    inputsVI = []

    outputs = []
    outputsVI = []

    tensorsToKeep = []

    nodesToRemove = []

    # Remove unwanted operators
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


    # Remove unused tensors
    keepOnlyTensors(model.graph.initializer, tensorsToKeep)
    keepOnlyTensors(model.graph.value_info, tensorsToKeep)


    # Find the model inputs and outputs
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

    # Assign new model inputs and outputs
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
    """ 
        Take the ONNX model in 'originalOnnxFile'. Reduce it to only contain
        operators with indices 'startIdx' to 'ednIdx' (both included). Save the 
        reduced model in 'outOnnxFile' and convert it to TFLite in 
        'outTfliteFile'. Then run both reduced models with the same random
        input data and print statistics. 
        
        Doesn't work when the ONNX model does not have internal tensor shapes 
        specified! For example TinyYOLO v2.
        """
    
    onnxModel = pickOutOperators(originalOnnxFile,startIdx, endIdx)          

    print(f"\tTesting from '{onnxModel.graph.node[0].op_type}' to",
          f"'{onnxModel.graph.node[endIdx-startIdx].op_type}'")
    onnx.save(onnxModel, outOnnxFile)
    convert.convertModel(outOnnxFile, outTfliteFile)

    # Get input shape
    shape = [dim.dim_value if dim.dim_value != 0 else 1 for dim in onnxModel.graph.input[0].type.tensor_type.shape.dim]
    shape = shapeToNHWC(shape)

    # Generate random input
    inpt: np.ndarray = np.random.rand(*shape).astype(np.float32)

    # Run the models and print stats
    onnxOut = runOnnxModel(outOnnxFile, inpt)
    printStats("ONNX", onnxOut)

    tflOut = tensorToNCHW(runTFLiteModel(outTfliteFile, inpt))
    printStats("TFLite", tflOut)

    diff = onnxOut - tflOut
    printStats("Difference", diff)


def testConversion(onnxFile, tflFile, numIterations):
    """ 
        MAIN TESTING FUNCTION
        
        Convert ONNX model in 'onnxFile' to TFLite and save it in 'tflFile'. 
        Then run both models with the same random data 'numIterations' times'.
        Finally print statistics of the outputs. 
    """
    
    onnxModel = onnx.load(onnxFile)

    convert.convertModel(onnxFile, tflFile)

    # Get input shape
    shape = [dim.dim_value if dim.dim_value != 0 else 1 for dim in onnxModel.graph.input[0].type.tensor_type.shape.dim]
    shape = shapeToNHWC(shape)

    # Run models with random inputs and collect statistics
    absErr = []
    relErr = []
    meanErr = []
    meanRelErr = []
    stdErr = []
    stdRelErr = []
    for i in range(numIterations):
        # Generate random input
        inpt: np.ndarray = np.random.rand(*shape).astype(np.float32)

        # Run models
        oOut = runOnnxModel(onnxFile, inpt)
        tOut = tensorToNCHW(runTFLiteModel(tflFile, inpt))

        # Collect statistics
        tmpAbsErr = np.abs(oOut - tOut)
        tmpPercErr = tmpAbsErr / np.abs(oOut)

        absErr.append(tmpAbsErr)
        relErr.append(tmpPercErr)

        tmpMeanErr = np.abs(oOut.mean() - tOut.mean())
        meanErr.append(tmpMeanErr)
        meanRelErr.append(tmpMeanErr / np.abs(oOut.mean()))

        tmpStdErr = np.abs(oOut.std() - tOut.std())
        stdErr.append(tmpStdErr)
        stdRelErr.append(tmpStdErr / np.abs(oOut.std()))


    # Calculate the mean and max absolute error
    maxSum = 0.0
    meanSum = 0.0
    for el in absErr:
        maxSum += el.max()
        meanSum += el.mean()
    print("Max Absolute error =\t", "%.6e" % (maxSum/numIterations))
    print("Mean Absolute error =\t", "%.6e" % (meanSum/numIterations))

    # Calculate the mean and max percentage error
    maxSum = 0.0
    meanSum = 0.0
    for el in relErr:
        maxSum += el.max()
        meanSum += el.mean()
    print("Max Relative error =\t", "%.6e" % (maxSum/numIterations))
    print("Mean Relative error =\t", "%.6e" % (meanSum/numIterations))

    # Calculate statistics
    print("Max Absolute error of output mean =\t", "%.6e" % np.asarray(meanErr).max())
    print("Average Absolute error of output mean =\t", "%.6e" % np.asarray(meanErr).mean())

    print("Max Relative error of output mean =\t", "%.6e" % np.asarray(meanRelErr).max())
    print("Average Relative error of output mean =\t", "%.6e" % np.asarray(meanRelErr).mean())

    print("Max Absolute error of output std =\t", "%.6e" % np.asarray(stdErr).max())
    print("Average Absolute error of output std =\t", "%.6e" % np.asarray(stdErr).mean())

    print("Max Relative error of output std =\t", "%.6e" % np.asarray(stdRelErr).max())
    print("Average Relative error of output std =\t", "%.6e" % np.asarray(stdRelErr).mean())




""" -------------------- Start of execution -------------------- """

imageFile = "data/224x224/cat2.jpg"
onnxFile = "data/onnx/ResNet101-DUC-12.onnx"
onnxReducedFile = "test/duc.onnx"
tflReducedFile = "test/duc.tflite"

alexnetOnnx = "data/onnx/bvlcalexnet-12.onnx"
alexnetTfl = "test/alexnet.tflite"

ducOnnx = "data/onnx/ResNet101-DUC-12.onnx"
ducTfl = "test/duc.tflite"

tinyyoloOnnx = "data/onnx/tinyyolov2-8.onnx"
tinyyoloTfl = "test/tinyyolo.tflite"

speechOnnx = "data/onnx/speech_command_classifier_trained.onnx"
speechTfl = "test/speech_command_classifier_trained.tflite"


""" ---------- Quick tests ---------- """

# # TEST ALEXNET CONVERSION
# print("\tTesting Alexnet conversion.")
# runAndTestOperators(alexnetOnnx, "test/alexnet.onnx", alexnetTfl, 0, 23)
# exit()

# # TEST TINYYOLO CONVERSION
# print("\tTesting TinyYOLO v2 conversion.")
# runAndTestOperators(tinyyoloOnnx, "test/tinyyolo.onnx", tinyyoloTfl, 0, 32)
# exit()

# # TEST RESNET-DUC CONVERSION
# print("\tTesting Resnet-DUC conversion.")
# runAndTestOperators(ducOnnx, "test/duc.onnx", ducTfl, 0, 354)
# exit()

# # TEST SPEECH CLASSIFIER CONVERSION
# print("\tTesting Speech Classifier conversion.")
# runAndTestOperators(speechOnnx, "test/duc.onnx",  speechTfl, 0, 17)
# exit()




""" ---------- Thorough tests ---------- """

# # TEST ALEXNET CONVERSION
# print("\tTesting Alexnet conversion.")
# testConversion(alexnetOnnx, alexnetTfl, 10)
# exit()

# # TEST TINYYOLO CONVERSION
# print("\tTesting TinyYOLO v2 conversion.")
# testConversion(tinyyoloOnnx, tinyyoloTfl, 10)
# exit()

# # TEST RESNET-DUC CONVERSION
# print("\tTesting Resnet-DUC conversion.")
# testConversion(ducOnnx, ducTfl, 1)
# exit()

# TEST SPEECH CLASSIFIER CONVERSION
print("\tTesting Speech Classifier conversion.")
testConversion(speechOnnx, speechTfl, 10)
exit()
