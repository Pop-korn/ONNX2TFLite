from typing import Dict

import lib.tflite.BuiltinOperator as tflBO

import src.generator.model.Model as tflM
import src.generator.model.SubGraphs as tflSG
import src.generator.model.Buffers as tflB
import src.generator.model.Tensors as tflT
import src.generator.model.Operators as tflO
import src.generator.model.OperatorCodes as tflOC

import src.parser.model.ValueInfo as onnxVI
import src.parser.model.Tensors as onnxT

import src.converter.conversion.Translator as Translator

import src.err as err

class ModelBuilder:
    """ This class provides methods to build a TFLite model by parts. """
    
    __tflModel: tflM.Model
    __tensorNameMap: Dict # Mapping 'str' to 'tflT.Tensor'
    __opCodeTypeIndexMap: Dict # Mapping 'tflBO.BuiltinOperator' to 'int'

    def __init__(self, modelVersion: int, modelDescription: str) -> None:
        self.__tflModel = tflM.Model(modelVersion,modelDescription)
        self.__opCodeTypeIndexMap = {}
        self.__tensorNameMap = {}



    def finish(self) -> tflM.Model:
        """ Finalize the TFLite model and return it. """

        # Assign each buffer its index
        for i, buffer in enumerate(self.getBuffers().vector):
            buffer.tmpIndex = i

        # Assign each tensor its index and its buffer index
        for i, tensor in enumerate(self.getTensors().vector):
            tensor.tmpIndex = i
            tensor.buffer = tensor.tmpBuffer.tmpIndex

        # Assign 'Outputs' and 'Inputs' their tensor inidces
        outputs = self.getSubgraph().outputs
        for tensor in outputs.tmpOutputs:
            outputs.append(tensor.tmpIndex)

        inputs = self.getSubgraph().inputs
        for tensor in inputs.tmpInputs:
            inputs.append(tensor.tmpIndex)

        # Assign each operator its inputs and outputs indices
        for operator in self.getSubgraph().operators.vector:
            for inputTensor in operator.tmpInputs:
                operator.inputs.append( inputTensor.tmpIndex )

            for outputTensor in operator.tmpOutputs:
                operator.outputs.append( outputTensor.tmpIndex )

        return self.__tflModel





    def __buildOperatorCode(self, opType: tflBO.BuiltinOperator):
        """ Add a new OperatorCode for given 'opType' to the 'operator_codes' vector. """
        opCode = tflOC.OperatorCode(opType)

        self.getOperatorCodes().append(opCode)


    def buildBuffer(self, oTensor: onnxT.Tensor) -> tflB.Buffer:
        """ Create a new 'buffer' object. Register it and add to the 'model.Buffers'. """
        buffer = tflB.Buffer()

        if oTensor.data is None:
            # No data was provided in the tensor
            err.warning(f"ONNX Tensor '{oTensor.name}' should contain data but doesn't! Generating empty buffer!")
            self.appendNewBuffer(buffer)
            return


        # Convert the data
        buffer.type = Translator.convertDataType(oTensor.dataType)
        buffer.data = Translator.convertTensorData(oTensor.data, oTensor.dims)

        self.appendNewBuffer(buffer)

        return buffer


    def buildConstantTensor(self, oTensor: onnxT.Tensor, buffer: tflB.Buffer):
        """ Create a 'Tensor' object from the ONNX 'oTensor'. Register it and add to the 
            'subGraph.Tensors'. """
        shape = Translator.convertShapeDims(oTensor.dims)
        name = oTensor.name
        type = Translator.convertDataType(oTensor.dataType)

        tTensor = tflT.Tensor(shape,name,None,type)
        tTensor.tmpBuffer = buffer

        self.appendNewTensor(tTensor)


    def buildEmptyTensor(self, oVI :onnxVI.ValueInfo, buffer: tflB.Buffer):
        """ Create a 'Tensor' object from am ONNX ValueInfo object. So the resulting tensor has
            no data, just properties. """
        oTensor = oVI.type.tensorType
        if oTensor is None:
            err.error(err.Code.UNSUPPORTED_ONNX_TYPE,"ONNX: Only type 'tensor_type' is supported yet!")

        shape = Translator.convertShape(oTensor.shape)
        name = oVI.name
        type = Translator.convertDataType(oTensor.elemType)

        tensor = tflT.Tensor(shape, name, None, type)
        tensor.tmpBuffer = buffer
        self.appendNewTensor(tensor)


    def buildEmptyBuffer(self) -> tflB.Buffer:
        """ Create, register and return a new empty 'Buffer' object. """
        buffer = tflB.Buffer([])

        self.getBuffers().append(buffer)

        return buffer



    """ -------------------- 'quality of life' functions. -------------------- """


    def opCodeIndexForOpType(self, opType: tflBO.BuiltinOperator):
        """ Return the index to the 'operator_codes' vector in the TFLite model for
            the operator with given 'opType'.
            If corresponding opCode doesn't exist, create new mapping and a new OperatorCode. """
        if opType not in self.__opCodeTypeIndexMap.keys():
            self.__opCodeTypeIndexMap[opType] = self.operatorCodesSize()
            self.__buildOperatorCode(opType)
        
        return self.__opCodeTypeIndexMap[opType]



    def tensorExists(self, name: str):
        """ Determine if a tensor with 'name' already exists or not. """
        return name in self.__tensorNameMap.keys()
    

    def tensorForName(self, name: str) -> tflT.Tensor:
        """ Get an existing TFLite tensor with given 'name'. If such tensor
            does NOT exist, function will create and register a new tensor with
            shape '[]', which will be returned."""
        if name not in self.__tensorNameMap.keys():
            err.note(f"Tensor '{name}' is not yet in the tensors. Adding it!") 

            newTensor = tflT.Tensor(tflT.Shape([]),name) # TODO Should be OK (only useless output tensor)
            newTensor.tmpBuffer = self.buildEmptyBuffer()

            self.appendNewTensor(newTensor)

        return self.__tensorNameMap[name]


    def bufferSize(self):
        """ Return the number of buffers that are currently in the model. """
        return self.getBuffers().len()



    def operatorCodesSize(self):
        """ Return the number of buffers that are currently in the model. """
        return len(self.__opCodeTypeIndexMap.keys())


    def appendNewTensor(self, tTensor: tflT.Tensor):
        """ Append the TFLite tensor 'tTensor' to the 'subGraph.tensors'
            and register it. """

        if tTensor.name in self.__tensorNameMap.keys():
            err.warning(f"Tensor '{tTensor.name}' is already in the tensors!")  
        else:
            self.__tensorNameMap[tTensor.name] = tTensor
            self.getTensors().append(tTensor)


    def appendNewBuffer(self, buffer: tflB.Buffer):
        """ Append the 'buffer' to the 'model.buffers'. """
        self.getBuffers().append(buffer)




    """ ---------------- Functions to get an element of the TFLite model. ----------------
    If the element doesn't exist, it is created. So functions always return a valid object. """


    def getSubgraphs(self) -> tflSG.SubGraphs:
        if self.__tflModel.subGraphs is None:
            self.__tflModel.subGraphs = tflSG.SubGraphs()

        return self.__tflModel.subGraphs


    def getSubgraph(self) -> tflSG.SubGraph:
        subGraphs = self.getSubgraphs()
        if subGraphs.len() == 0:
            subGraphs.append(tflSG.SubGraph())

        return subGraphs.get(0)


    def getTensors(self) -> tflT.Tensors:
        subGraph = self.getSubgraph()
        if subGraph.tensors is None:
            subGraph.tensors = tflT.Tensors()
        
        return subGraph.tensors


    def getBuffers(self) -> tflB.Buffers:
        if self.__tflModel.buffers is None:
            self.__tflModel.buffers = tflB.Buffers()

        return self.__tflModel.buffers

    
    def getOperators(self) -> tflO.Operators:
        subGraph = self.getSubgraph()
        if subGraph.operators is None:
            subGraph.operators = tflO.Operators()

        return subGraph.operators

    def getOperatorCodes(self) -> tflOC.OperatorCodes:
        if self.__tflModel.operatorCodes is None:
            self.__tflModel.operatorCodes = tflOC.OperatorCodes()

        return self.__tflModel.operatorCodes
