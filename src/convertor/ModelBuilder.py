import src.generator.model.Model as tflM
import src.generator.model.SubGraphs as tflSG
import src.generator.model.Buffers as tflB
import src.generator.model.Tensors as tflT

import src.parser.model.ValueInfo as onnxVI

import src.convertor.Convertor as Convertor

import src.err as err

class Builder:
    __tflModel: tflM.Model
    __bufferNameIndexMap: dict[str : int]

    def __init__(self, modelVersion: int) -> None:
        self.__tflModel = tflM.Model(modelVersion)
        self.__bufferNameIndexMap = {}

    def buildOutputTensors(self, oOutputs: list[onnxVI.ValueInfo]):
        for oOutput in oOutputs:
            if oOutput.type.tensorType is None:
                err.eprint(err.Code.UNSUPPORTED_ONNX_TYPE,"ONNX: Only type 'tensor_type' is supported yet!")

            self.__buildEmptyTensor(oOutput)


    def finish(self) -> tflM.Model:
        return self.__tflModel


    """ -------------------- Private 'quality of life' functions. -------------------- """


    def __bufferIndexForName(self, name: str):
        """ Return the index to the 'buffers' vector in the TFLite model for the tensor with
            given name """
        if name not in self.__bufferNameIndexMap.keys():
            self.__bufferNameIndexMap[name] = len(self.__bufferNameIndexMap.keys())

        return self.__bufferNameIndexMap[name]


    """ -------------------- Private generic build functions. -------------------- """
    

    def __buildEmptyTensor(self, oVI :onnxVI.ValueInfo):
        """ Build a 'Tensor' object from am ONNX ValueInfo object. So the resulting tensor has
            no data, just properties. """
        oTensor = oVI.type.tensorType
        if oTensor is None:
            err.eprint(err.Code.UNSUPPORTED_ONNX_TYPE,"ONNX: Only type 'tensor_type' is supported yet!")

        shape = Convertor.convertShape(oTensor.shape, keepDims=True)
        name = oVI.name
        bufferIndex = self.__bufferIndexForName(name)

        tensor = tflT.Tensor(shape, name, bufferIndex)
        self.__getTensors().append(tensor)

        self.__buildEmptyBuffer()

    def __buildEmptyBuffer(self):
        buffers = self.__getBuffers()
        buffers.append(tflB.Buffer([]))


    """ ---------------- Private functions to get an element of the TFLite model. ----------------
     If the element doesn't exist, it is created. So functions always return a valid object. """

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
        
