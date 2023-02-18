import src.convertor.tensors.TensorConvertor as tc

import src.generator.model.Model as tflModel

import src.parser.model.Model as onnxModel # onnxModel
    

def convertModel(oM: onnxModel.Model) -> tflModel.Model:

    tM = tflModel.Model()

