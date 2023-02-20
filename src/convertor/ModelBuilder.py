import src.generator.model.Model as tflM
import src.generator.model.SubGraphs as tflSG
import src.generator.model.Buffers as tflB
import src.generator.model.Tensors as tflT

import src.parser.model.ValueInfo as onnxVI
import src.parser.model.Tensors as onnxT

import src.convertor.Convertor as Convertor

import src.err as err

class Builder:
    __tflModel: tflM.Model
    __bufferNameIndexMap: dict[str : int]
    __tensorNameIndexMap: dict[str : int]

    def __init__(self, modelVersion: int, modelDescription: str) -> None:
        self.__tflModel = tflM.Model(modelVersion,modelDescription)
        self.__bufferNameIndexMap = {}
        self.__tensorNameIndexMap = {}

    def buildConstantTensors(self, oTensors: onnxT.Tensors):
        for oTensor in oTensors:
            pass

            

    def buildOutputTensors(self, oOutputs: list[onnxVI.ValueInfo]):
        """ Create 'tensor' tables in the 'tensors' vector of the subGraph for the 'oOutputs'.
            Also create empty buffers in the 'buffers' vector of the model. """

        outputs = tflSG.Outputs()

        for oOutput in oOutputs:
            if oOutput.type.tensorType is None:
                err.eprint(err.Code.UNSUPPORTED_ONNX_TYPE,"ONNX: Only type 'tensor_type' is supported for Outputs yet!")

            self.__buildEmptyBuffer(oOutput.name)
            self.__buildEmptyTensor(oOutput)

            outputs.append(self.__tensorIndexForName(oOutput.name))

        self.__getSubgraph().outputs = outputs

    def buildInputTensors(self, oInputs: list[onnxVI.ValueInfo]):
        """ Create 'tensor' tables in the 'tensors' vector of the subGraph for the 'oInputs'.
            Also create empty buffers in the 'buffers' vector of the model. """

        inputs = tflSG.Inputs()

        for oInput in oInputs:
            if oInput.type.tensorType is None:
                err.eprint(err.Code.UNSUPPORTED_ONNX_TYPE,"ONNX: Only type 'tensor_type' is supported for Inputs yet!")

            self.__buildEmptyBuffer(oInput.name)
            self.__buildEmptyTensor(oInput)

            inputs.append(self.__tensorIndexForName(oInput.name))

        self.__getSubgraph().inputs = inputs


    def finish(self) -> tflM.Model:
        return self.__tflModel


    """ -------------------- Private 'quality of life' functions. -------------------- """


    def __bufferIndexForName(self, name: str):
        """ Return the index to the 'buffers' vector in the TFLite model for the tensor with
            given name.
            If 'name' is not yet in the 'buffers', mapping will be added and warning will be printed. """
        if name not in self.__bufferNameIndexMap.keys():
            self.__bufferNameIndexMap[name] = self.__bufferSize()
            err.wprint(f"Tensor '{name}' is not yet in the buffer. Adding it on index '{self.__bufferNameIndexMap[name]}!'") 

        return self.__bufferNameIndexMap[name]

    def __tensorIndexForName(self, name: str):
        """ Return the index to the 'tensors' vector in the TFLite subGraph for the tensor with
            given name 
            If 'name' is not yet in the 'tensors', mapping will be added and warning will be printed. """
        if name not in self.__tensorNameIndexMap.keys():
            self.__tensorNameIndexMap[name] = self.__tensorsSize()
            err.wprint(f"Tensor '{name}' is not yet in the tensors. Adding it on index '{self.__bufferNameIndexMap[name]}!'") 


        return self.__tensorNameIndexMap[name]

    def __bufferSize(self):
        """ Return the number of buffers that are currently in the model. """
        return len(self.__bufferNameIndexMap.keys())

    def __tensorsSize(self):
        """ Return the number of tensors that are currently in the subGraph. """
        return len(self.__tensorNameIndexMap.keys())

    def __newBufferIndexForName(self, name: str):
        """ Return the index to the 'buffers' vector in the TFLite model for the tensor with
            given name. Just like in '__bufferIndexForName'.
            Howerver if 'name' is already in, warning message will be printed."""
        if name in self.__bufferNameIndexMap.keys():
            err.wprint(f"Tensor '{name}' is already in the buffer on index '{self.__bufferNameIndexMap[name]}!'")   
        else:
            # Add the new tensor
            self.__bufferNameIndexMap[name] =self.__bufferSize()

        return self.__bufferNameIndexMap[name]

    def __newTensorIndexForName(self, name: str):
        """ Return the index to the 'tensors' vector in the TFLite subGraph for the tensor with
            given name. Just like in '__tensorIndexForName'.
            Howerver if 'name' is already in, warning message will be printed."""
        if name in self.__tensorNameIndexMap.keys():
            err.wprint(f"Tensor '{name}' is already in the tensors on index '{self.__tensorNameIndexMap[name]}!'")   
        else:
            # Add the new tensor
            self.__tensorNameIndexMap[name] =self.__tensorsSize()

        return self.__tensorNameIndexMap[name]
               

    def __appendNewTensor(self, tTensor: tflT.Tensor):
        """ Add 'tTensor' to the end of the 'Tensors' vector.
            Function also assigns new index to the 'Tensors' vector for 'tTensor'. """
        self.__newTensorIndexForName(tTensor.name) # Register the new tensor 
        self.__getTensors().append(tTensor)

    def __appendNewBuffer(self, buffer: tflB.Buffer, name: str):
        """ Add 'buffer' to the end of the 'Buffers' vector.
            Function also assigns new index to the 'Buffers' vector for 'buffer'. 
            'name' is the name of the Tensor, the 'buffer' belongs to. """
        self.__newBufferIndexForName(name) # Register the new tensor 
        self.__getBuffers().append(buffer)


    """ -------------------- Private generic build functions. -------------------- """
    

    def __buildConstantTensor(self, oTensor: onnxT.Tensor):
            shape = Convertor.convertShapeDims(oTensor.dims)
            name = oTensor.name
            bufferIndex = self.__bufferIndexForName(name)
            type = Convertor.convertDataType(oTensor.dataType)

            tTensor = tflT.Tensor(shape,name,bufferIndex,type)
            self.__appendNewTensor(tTensor)


    def __buildEmptyTensor(self, oVI :onnxVI.ValueInfo):
        """ Build a 'Tensor' object from am ONNX ValueInfo object. So the resulting tensor has
            no data, just properties. """
        oTensor = oVI.type.tensorType
        if oTensor is None:
            err.eprint(err.Code.UNSUPPORTED_ONNX_TYPE,"ONNX: Only type 'tensor_type' is supported yet!")

        shape = Convertor.convertShape(oTensor.shape)
        name = oVI.name
        bufferIndex = self.__bufferIndexForName(name)
        type = Convertor.convertDataType(oTensor.elemType)

        tensor = tflT.Tensor(shape, name, bufferIndex, type)
        self.__appendNewTensor(tensor)

    def __buildEmptyBuffer(self, name: str):
        """ Add an empty buffer to the 'buffers' vector and map 'name' to the corresponding index. 
            'name' is the name of the Tensor, the 'buffer' belongs to. """
        buffers = self.__getBuffers()
        buffers.append(tflB.Buffer([]))
        self.__newBufferIndexForName(name)


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
        
