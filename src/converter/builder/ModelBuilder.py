from typing import Dict, List

import numpy as np

from lib.tflite import (
    BuiltinOperator as tflBO,
    BuiltinOptions as tflBOpt,
    TensorType as tflTT,
    ActivationFunctionType as tflAFT
)

from src.generator.model import (
    Model as tflM,
    SubGraphs as tflSG,
    Buffers as tflB,
    Tensors as tflT,
    Operators as tflO,
    OperatorCodes as tflOC
)

from src.parser.model import (
    ValueInfo as onnxVI,
    Tensors as onnxT
)

from src.generator.builtin import (
    Transpose as tflTranspose
)

import src.converter.conversion.Translator as Translator

import src.err as err

class ModelBuilder:
    """ This class provides methods to build a TFLite model by parts. """
    
    __tflModel: tflM.Model

    __tensorNameMap: Dict # Mapping 'str' to 'tflT.Tensor'

    __opCodeTypeIndexMap: Dict # Mapping 'tflBO.BuiltinOperator' to 'int'

    __nchwTensorVersion: Dict # Mapping 'tflT.Tensor' to 'tflT.Tensor' which is
                              # equal, but in NCHW format

    __skippedOutputMap: Dict # Mapping 'tflT.Tensor' objects that were outputs
                             # of skipped operators, to 'tflT.Tensor' ouputs of 
                             # previous operators


    def __init__(self, modelVersion: int, modelDescription: str) -> None:
        self.__tflModel = tflM.Model(modelVersion, modelDescription)
        self.__opCodeTypeIndexMap = {}
        self.__tensorNameMap = {}
        self.__nchwTensorVersion = {}
        self.__skippedOutputMap = {}


    def nchwVersionOf(self,tTensor: tflT.Tensor):
        """ Get the NCHW version of non-static 'tTensor'. If one is not 
            available in the graph yet, add transpose operator to create it. """
        if tTensor in self.__nchwTensorVersion.keys():
            return self.__nchwTensorVersion[tTensor]
        
        # Need to add Transpose operator to transform 'tTensor' to NCHW.

        nchwTensor = self.duplicateTensor(tTensor, tTensor.name + "_nchw")
        nchwTensor.shape = Translator.NHWCShapeToNCHW(tTensor.shape)

        perm = Translator.createToNCHWPerm(tTensor.shape.vector)

        shapeTensor = self.__createTensorForData(perm,
                                                 "transpose_to_nchw_perm")

        transpose = tflO.Operator(builtinOptions=tflTranspose.Transpose())
        transpose.opcodeIndex = self.opCodeIndexForOpType(transpose.builtinOptions.operatorType)
        transpose.tmpInputs = [tTensor, shapeTensor]
        transpose.tmpOutputs = [nchwTensor]

        self.checkAndAppendOperator(transpose)

        self.__nchwTensorVersion[tTensor] = nchwTensor

        return nchwTensor


    def skipOperator(self, tOp: tflO.Operator):
        """ Map the outputs of 'tOp' to the inputs of 'tOp'. Future references 
            to the outputs of 'tOp' will be substituted for its inputs. This is
            done in the 'checkAndAppendOperator()' method. """

        # Try to find suitable replacements
        for skipped in tOp.tmpOutputs:
            if skipped in self.__skippedOutputMap.keys():
                err.internal(f"skipOperator: tensor '{skipped.name}' is already"
                             , "mapped to something!")
                continue

            for replacement in tOp.tmpInputs:
                if self.tensorHasData(replacement):
                    # Not a dynamic tensor
                    continue

                # Found suitable replacement
                self.__skippedOutputMap[skipped] = replacement
                print("Replacing",skipped.name,"with", replacement.name)

                # Check if we are skipping the output of the whole graph
                graphOutputs = self.getSubgraph().outputs.tmpOutputs 
                if skipped in graphOutputs:
                    idx = graphOutputs.index(skipped)
                    graphOutputs[idx] = replacement

                # Don't look for more replacements
                break


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

    
    def createTransposedTensor(self, tTensor: tflT.Tensor) -> tflT.Tensor:
        """ Create a new tensor and buffer, which is exactly the same as 
            'tTensor' except its data is transposed and name is changed. 
            Return the transposed tensor. """
        
        err.note(f"Creating a transposed tensor for '{tTensor.name}'.")

        newTensor = self.duplicateTensor(tTensor, tTensor.name + "_transposed")

        # TODO needs testing
        newTensor.tmpBuffer.data = np.transpose(newTensor.tmpBuffer.data)
        newTensor.shape = tflT.Shape(list(newTensor.tmpBuffer.data.shape))

        return newTensor


    def duplicateTensor(self, tTensor: tflT.Tensor, 
                        newName: str) -> tflT.Tensor:
        """ Properly create and register a new TFLite tensor, add it to the 
            model and return it. 
            The new tensors 'newName' will be used, unless a tensor with that
            name already exists. In which case the name will be slightelly 
            altered. """
        
        newName = self.__validateNewTensorName(newName)

        buffer = tflB.Buffer()
        if self.tensorHasData(tTensor):
            buffer.data = tTensor.tmpBuffer.data.copy()

        self.appendNewBuffer(buffer)

        shape = tflT.Shape(tTensor.shape.vector.copy())
        tensor = tflT.Tensor(shape, newName, None, tTensor.type,
                             tTensor.quantization, tTensor.isVariable,
                             tTensor.hasRank)
        tensor.tmpBuffer = buffer

        self.appendNewTensor(tensor)

        return tensor
        

    def finish(self) -> tflM.Model:
        """ Finalize the TFLite model and return it. """

        # Keep only 1 empty buffer
        self.__redirectEmptyTensorsToOneBuffer()

        # Combine activation function operators where possible
        self.__fuseActivationFunctions()

        # Remove unused tensors and bufers
        self.__removeUnusedTensorsAndBuffers()

        # Swich from using 'tmp' references to 'index' references in tensors
        # and buffers.
        self.__assignTensorAndBufferIndices()

        return self.__tflModel
    

    def __fuseActivationFunctions(self):
        if tflBO.BuiltinOperator.RELU not in self.__opCodeTypeIndexMap.keys():
            # There is no RELU operator in the model
            return
        
        reluOpCodeIndex = self.opCodeIndexForOpType(tflBO.BuiltinOperator.RELU)

        self.__countReferencesToTensorsAndBuffers()

        # Find Relu operators and remove them if possible
        for op in self.getOperators().vector:
            
            try:

                if op.opcodeIndex != reluOpCodeIndex:
                    # Operator is not Relu
                    continue
                
                inTensor = op.tmpInputs[0]
                
                if inTensor.tmpReferenceCount != 2:
                    # The output of the previous tensor is used by something 
                    # other that the Relu.
                    continue

                prevOp = self.__getOperatorWithOutput(inTensor)

                if not self.__canHaveFusedActivationFunction(prevOp):
                    # The operator before Relu does not support fused act. fun.
                    continue

                if prevOp.tmpOutputs[0] != inTensor:
                    # The Relu is being applied to a different output
                    # than the main one. Keep things as they are
                    continue

                if not self.tensorsSimilar(inTensor, op.tmpOutputs[0]):
                    # Should never happen
                    err.internal("ModelBuilder.__fuseActivationFunctions()",
                                 "Relu input differs from its output!")
                    continue

                if prevOp.builtinOptions.fusedActivationFunction != tflAFT.ActivationFunctionType.NONE:
                    # Previous operator already has an activation function
                    continue

                # Finally fuse the activation function with 'prevOp'
                prevOp.tmpOutputs[0] = op.tmpOutputs[0]
                prevOp.builtinOptions.fusedActivationFunction = tflAFT.ActivationFunctionType.RELU
                self.getOperators().remove(op)

            except:
                err.note("Something failed during activation function fusing.")


    def __assignTensorAndBufferIndices(self):
        """ Correctly initialize all references via indices in all tensors
            and buffers. """
        
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


    def __redirectEmptyTensorsToOneBuffer(self):
        emptyBuffer = self.__getFirstEmptyBuffer()

        for t in self.getTensors().vector:
            if not self.tensorHasData(t):
                t.tmpBuffer = emptyBuffer


    def __removeUnusedTensorsAndBuffers(self):
        """ Remove all tensors and buffers from the model, that are not
            marked as used. """
        
        self.__countReferencesToTensorsAndBuffers()

        toRemove = []
        for tensor in self.getTensors().vector:
            if tensor.tmpReferenceCount == 0:
                toRemove.append(tensor)
        for tensor in toRemove:
            self.getTensors().remove(tensor)

        toRemove = []
        for buffer in self.getBuffers().vector:
            if buffer.tmpReferenceCount == 0:
                toRemove.append(buffer)
        for buffer in toRemove:
            self.getBuffers().remove(buffer)


    def __countReferencesToTensorsAndBuffers(self):
        """ Count how many times each tensor and buffer are used. Store
            the counts in their '.tmpReferenceCount' attributes. """
        
        # Mark all unused
        for tensor in self.getTensors().vector:
            tensor.tmpReferenceCount = 0

        for buffer in self.getBuffers().vector:
            buffer.tmpReferenceCount = 0

        # Find out which are used
        for operator in self.getOperators().vector:
            for tensor in operator.tmpInputs:
                tensor.tmpReferenceCount += 1
                if tensor.tmpBuffer is not None:
                    tensor.tmpBuffer.tmpReferenceCount += 1

            for tensor in operator.tmpOutputs:
                tensor.tmpReferenceCount += 1
                if tensor.tmpBuffer is not None:
                    tensor.tmpBuffer.tmpReferenceCount += 1


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

        # Append the tensor. Override=True, because sometimes the tensor might
        # have been mentioned as 'value_info', in which case it was added 
        # without data.
        self.appendNewTensor(tTensor, overwrite=True)


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
        buffer = tflB.Buffer()

        self.getBuffers().append(buffer)

        return buffer




    """ -------------------- 'quality of life' functions. -------------------- """


    def __canHaveFusedActivationFunction(self, tOp: tflO.Operator) -> bool:
        """ Determine if given operator can have a fused activation function
            as one of its parameters. """
        
        supportedOptions = [tflBOpt.BuiltinOptions.Conv2DOptions,
                            tflBOpt.BuiltinOptions.Conv3DOptions,
                            tflBOpt.BuiltinOptions.Pool2DOptions,
                            tflBOpt.BuiltinOptions.DepthwiseConv2DOptions,
                            tflBOpt.BuiltinOptions.SVDFOptions,
                            tflBOpt.BuiltinOptions.RNNOptions,
                            tflBOpt.BuiltinOptions.SequenceRNNOptions,
                            tflBOpt.BuiltinOptions.BidirectionalSequenceRNNOptions,
                            tflBOpt.BuiltinOptions.FullyConnectedOptions,
                            tflBOpt.BuiltinOptions.ConcatenationOptions,
                            tflBOpt.BuiltinOptions.MulOptions,
                            tflBOpt.BuiltinOptions.L2NormOptions,
                            tflBOpt.BuiltinOptions.LSTMOptions,
                            tflBOpt.BuiltinOptions.UnidirectionalSequenceLSTMOptions,
                            tflBOpt.BuiltinOptions.SubOptions,
                            tflBOpt.BuiltinOptions.DivOptions,
                            tflBOpt.BuiltinOptions.TransposeOptions]
        
        return tOp.builtinOptions.builtinOptionsType in supportedOptions


    def __createTensorForData(self, data: np.ndarray, 
                              name: str):
        type = Translator.numpyTypeToTFLite(data.dtype)

        buffer = tflB.Buffer(data, type)
        self.appendNewBuffer(buffer)

        shape = Translator.shapeFromNumpy(data)
        name = self.__validateNewTensorName(name)

        tensor = tflT.Tensor(shape, name, type=type)

        tensor.tmpBuffer = buffer

        self.appendNewTensor(tensor)

        return tensor


    def __validateNewTensorName(self, name: str) -> str:
        """ Take tensor name 'name' and make it unique in the model.
            Returns a unique tensor name. """
        
        # Try adding numbers to the 'name' until it is unique
        suffix = 0
        newName = name
        while self.tensorExists(newName):
            newName = name + str(suffix)
            suffix += 1

        return newName
            

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


    def __removeTensorWithNameFromCollection(self, name, collection):
        """ Find and remove a tensor with given 'name' from given 'collection'.
        """
        toRemove = None

        for t in collection:
            if t.name == name:
                toRemove = t
                break

        if toRemove is not None:
            collection.remove(toRemove)


    def __removeInputWithName(self, name):
        """ Find and remove a tensor in the subgraph 'inputs' with given 'name'. 
        """
        self.__removeTensorWithNameFromCollection(name, 
                                                  self.getSubgraph().inputs.tmpInputs)
    
    
    def __removeOutputWithName(self, name):
        """ Find and remove a tensor in the subgraph 'outputs' with given 'name'. 
        """
        self.__removeTensorWithNameFromCollection(name, 
                                                  self.getSubgraph().outputs.tmpOutputs)
    
    
    def __removeTensorWithName(self, name):
        """ Find and remove a tensor in the graph with given 'name'. """
        self.__removeTensorWithNameFromCollection(name, 
                                                  self.getTensors().vector)


    def appendNewTensor(self, tTensor: tflT.Tensor, overwrite: bool = False):
        """ Append the TFLite tensor 'tTensor' to the 'subGraph.tensors'
            and register it. """

        if tTensor.name in self.__tensorNameMap.keys():
            """ Tensor has already been added. Sometimes however, ONNX models 
                will have tensors in their 'inputs' or 'outpus', which don't
                belong there and are in fact static. I this case we need to 
                overwrite the existing tensors. """
            
            if overwrite:
                self.__removeTensorWithName(tTensor.name)

                # If the tenor previously appeared in ONNX 'inputs' or 'outputs',
                # the old version MUST be removed from there.
                self.__removeInputWithName(tTensor.name)
                self.__removeOutputWithName(tTensor.name)

                self.getTensors().append(tTensor)
                self.__tensorNameMap[tTensor.name] = tTensor
            else:
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
    

    def tensorHasData(self, tTensor: tflT.Tensor) -> bool:
        """ Determine if given TFLite tensor has any data. """

        if tTensor.tmpBuffer is None:
            return False
        
        res = self.bufferHasData(tTensor.tmpBuffer)
        if res is None:
            res = False
            print(tTensor.name)

        return res
        
    
    def bufferHasData(self, tBuffer: tflB.Buffer) -> bool:
        """ Determine if given buffer has any data in it. """

        try:
            if tBuffer.data is None:
                return False
            
            size = tBuffer.data.size                
            return size != 0
        
        except Exception as e:
            err.internal("'ModelBuilder.bufferHasData()' failed!")
            print(e)
            return None


    def __getLastOperator(self) -> tflO.Operator:
        """ Get the last operator in the subGraphs 'operators' list. 
            Or None if the list is empty. """
        return self.getOperators().getLast()


    def __getFirstEmptyBuffer(self) -> tflB.Buffer:
        """ Return the first empty buffer in the model.
            It should be the one on index 0. """
        for b in self.getBuffers().vector:
            if not self.bufferHasData(b):
                return b
            
        # Execution should not reach this
        err.internal("ModeLBuilder.__getFirstEmptyBuffer()")


    def __getOperatorWithOutput(self, tTensor: tflT.Tensor) -> tflO.Operator:
        """ Get the first operator from the graph, that has 'tTensor' in its
            'tmpOutputs' list. 
            If such operator doesn't exist, return None. """
        
        for op in self.getOperators().vector:
            if tTensor in op.tmpOutputs:
                return op
            
        return None
    

    def __getOperatorWithInput(self, tTensor: tflT.Tensor) -> tflO.Operator:
        """ Get the first operator from the graph, that has 'tTensor' in its
            'tmpInputs' list. 
            If such operator doesn't exist, return None. """
        
        for op in self.getOperators().vector:
            if tTensor in op.tmpInputs:
                return op
            
        return None


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
    