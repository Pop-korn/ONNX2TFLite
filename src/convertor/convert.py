import src.convertor.tensors.TensorConvertor as tc

import src.generator.model.Model as tflModel
import src.generator.model.Tensors as tflTnsors

import src.parser.model.Model as onnxModel # onnxModel

def convertTensors(oM: onnxModel.Model):

    tTensors = tflTnsors.Tensors()

    for tensor in oM.graph.initializers:
        print(tensor.dims)

    

def convertModel(oM: onnxModel.Model) -> tflModel.Model:

    tM = tflModel.Model()

    convertTensors(oM.graph.initializers)
