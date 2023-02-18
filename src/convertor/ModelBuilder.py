import src.generator.model.Model as tflM
import src.generator.model.SubGraphs as tflSG
import src.generator.model.Buffers as tflB
import src.generator.model.Tensors as tflT

import src.parser.model.ValueInfo as onnxVI

import src.convertor.Convertor as Convertor

import src.err as err

class Builder:
    __tflModel: tflM.Model

    def __init__(self, modelVersion: int) -> None:
        self.__tflModel = tflM.Model(modelVersion)

    def buildOutputTensors(self, oOutputs: list[onnxVI.ValueInfo]):
        subGraph = self.__getSubgraph()

        for oOutput in oOutputs:
            if oOutput.type.tensorType is None:
                err.eprint(err.Code.UNSUPPORTED_ONNX_TYPE,"ONNX: Only type 'tensor_type' is supported yet!")

            self.__buildTensor(oOutput)


    def finish(self) -> tflM.Model:
        return self.__tflModel

    
    def __buildTensor(self, oVI :onnxVI.ValueInfo):
        oTensor = oVI.type.tensorType
        if oTensor is None:
            err.eprint(err.Code.UNSUPPORTED_ONNX_TYPE,"ONNX: Only type 'tensor_type' is supported yet!")

        shape = Convertor.convertShape(oTensor.shape)
        name = oVI.name

        tensor = tflT.Tensor(shape, name)

        tensors = self.__getTensors()

        tensors.append(tensor)

    """ Private functions to get an element of the TFLite model. If the element doesn't exist,
        it is created. So functions always return a valid object. """

    def __getSubgraphs(self) -> tflSG.SubGraphs:
        if self.__tflModel.subGraphs is None:
            self.__tflModel.subGraphs = tflSG.SubGraphs()

        return self.__tflModel.subGraphs

    def __getSubgraph(self) -> tflSG.SubGraph:
        subGraphs = self.__getSubgraphs()
        if subGraphs.len() == 0:
            subGraphs.append(tflSG.SubGraph())

        return subGraphs.get(0)

    def __getTensors(self) -> tflT.Tensors:
        subGraph = self.__getSubgraph()
        if subGraph.tensors is None:
            subGraph.tensors = tflT.Tensors()
        
        return subGraph.tensors

    def __getBuffers(self) -> tflB.Buffers:
        if self.__tflModel.buffers is None:
            self.__tflModel.buffers = tflB.Buffers()

        return self.__tflModel.buffers
        
