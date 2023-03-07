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
    __skippedOutputMap: Dict # Mapping 'tflT.Tensor' objects that were
    # outputs of skipped operators, to 'tflT.Tensor' ouputs of previous operators


    def __init__(self, modelVersion: int, modelDescription: str) -> None:
        self.__tflModel = tflM.Model(modelVersion,modelDescription)
        self.__opCodeTypeIndexMap = {}
        self.__tensorNameMap = {}
        self.__skippedOutputMap = {}


    def skipOperator(self, tOp: tflO.Operator):
        """ Map the outputs of 'tOp' to the outputs of the previous operator,
            or the graph input. """
        
        lastOperator = self.__getLastOperator()
        if lastOperator is None:
            # The very first operator is being skipped
            lastOutputs = self.getSubgraph().inputs.tmpInputs
            err.unchecked("ModelBuilder.skipOperator(): first operator.")
        else:
            lastOutputs = lastOperator.tmpOutputs

        for skipped, replacement in zip(tOp.tmpOutputs, lastOutputs):
            self.__skippedOutputMap[skipped] = replacement



    def checkAndAppendOperator(self, tOp: tflO.Operator):
        """ Append the new TFLite operator the the model. Check that it's input
            tensors are valid.
            For example if conversion of the last operator was skipped, this 
            operator will be correctly connetced with the one before that. """
        
        # Find out if operators inputs were skipped
        for i, inpt in enumerate(tOp.tmpInputs):
            if inpt in self.__skippedOutputMap.keys():
                tOp.tmpInputs[i] = self.__skippedTensorReplacement(inpt)
            
                if not self.tensorsSimilar(tOp.tmpInputs[i], inpt):
                    # Tensors can not be replaced
                    err.error(None,f"Could not connect graph after operator was",
                            f"skipped! Tensor '{tOp.tmpInputs[i].name}'"
                            f"cannot match input '{inpt.name}'!")

        self.getOperators().append(tOp)



    def finish(self) -> tflM.Model:
        """ Finalize the TFLite model and return it. """

        # Remove unused tensors and bufers
        self.__markUnusedTensorsAndBuffers()
        self.__removeUnusedTensorsAndBuffers()

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
    

    def __removeUnusedTensorsAndBuffers(self):
        """ Remove all tensors and buffers from the model, that are not
            marked as used. """
        
        toRemove = []
        for tensor in self.getTensors().vector:
            if not tensor.tmpUsed:
                toRemove.append(tensor)
        for tensor in toRemove:
            self.getTensors().remove(tensor)

        toRemove = []
        for buffer in self.getBuffers().vector:
            if buffer.tmpUsed:
                toRemove.append(buffer)
        for buffer in toRemove:
            self.getTensors().remove(buffer)

    def __markUnusedTensorsAndBuffers(self):
        """ Find out which tensors and buffer in the model are actually used.
            Those that are not, will be marked by setting their 'tmpUsed'
            attribute to False. """
        
        # Mark all unused
        for tensor in self.getTensors().vector:
            tensor.tmpUsed = False
        for buffer in self.getBuffers().vector:
            buffer.tmpUsed = False

        # Find out which are used
        for operator in self.getOperators().vector:
            for tensor in operator.tmpInputs:
                tensor.tmpUsed = True
                if tensor.tmpBuffer is not None:
                    tensor.tmpBuffer.used = True

            for tensor in operator.tmpOutputs:
                tensor.tmpUsed = True
                if tensor.tmpBuffer is not None:
                    tensor.tmpBuffer.used = True


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
    

    def __skippedTensorReplacement(self, tTensor: tflT.Tensor):
        if tTensor not in self.__skippedOutputMap.keys():
            err.internal(f"Tensor '{tTensor.name}' was not skipped! But is trying",
                         "to be replaced!")
        return self.__skippedOutputMap[tTensor]


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


    def tensorsSimilar(self, tTensor1: tflT.Tensor, tTensor2: tflT.Tensor) -> bool:
        """ Determine if the given TFLite tensors have the same shape and 
            datatype. """
        
        if tTensor1.type != tTensor2.type:
            return False
        
        return Translator.collectionsEqual(tTensor1.shape.vector, 
                                           tTensor2.shape.vector)




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
    
    def __getLastOperator(self) -> tflO.Operator:
        """ Get the last operator in the subGraphs 'operators' list. 
            Or None if the list is empty. """
        return self.getOperators().getLast()
