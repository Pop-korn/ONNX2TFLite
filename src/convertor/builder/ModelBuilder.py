import lib.tflite.BuiltinOperator as tflBO

import src.generator.model.Model as tflM
import src.generator.model.SubGraphs as tflSG
import src.generator.model.Buffers as tflB
import src.generator.model.Tensors as tflT
import src.generator.model.Operators as tflO
import src.generator.model.OperatorCodes as tflOC

import src.parser.model.ValueInfo as onnxVI
import src.parser.model.Tensors as onnxT
import src.parser.model.Nodes as onnxN

import src.convertor.conversion.Convertor as Convertor
import src.convertor.conversion.OperatorConverter as opConvertor

import src.err as err

class ModelBuilder:
    """ This class provides methods to build a TFLite model by parts.
        Uses functions defined in the 'TensorBuilder' and 'OperatorBuilder'. """
    __tflModel: tflM.Model
    __tensorNameMap: dict[str : tflT.Tensor]
    __opCodeTypeIndexMap: dict[tflBO.BuiltinOperator : int]

    def __init__(self, modelVersion: int, modelDescription: str) -> None:
        self.__tflModel = tflM.Model(modelVersion,modelDescription)
        self.__opCodeTypeIndexMap = {}
        self.__tensorNameMap = {}

    


    """ -------------------- Public Builder functions -------------------- """


    def buildOperator(self, oNode:onnxN.Node):
        """ Convert an ONNX Node to a corresponding TFLite operator.
            This is ALWAYS a 1 to 1 conversion. """

        tOp = opConvertor.convertNode(oNode, self.__tensorForName)

        match(oNode.opType):
            case "Conv":
                tOp.builtinOptions, opCode = opConvertor.convertConv(oNode.attributes)
                tOp.opcodeIndex = self.__opCodeIndexForOpType(opCode)
            case "LRN":
                tOp.builtinOptions, opCode = opConvertor.convertLRN(oNode.attributes)
                tOp.opcodeIndex = self.__opCodeIndexForOpType(opCode)
            case "MaxPool":
                tOp.builtinOptions, opCode = opConvertor.convertMaxPool(oNode.attributes)
                tOp.opcodeIndex = self.__opCodeIndexForOpType(opCode)
            case "Relu":
                tOp.builtinOptions = None
                tOp.opcodeIndex = self.__opCodeIndexForOpType(tflBO.BuiltinOperator.RELU)
            # case "Reshape":
            #     tOp.inputs.vector.pop() # An extra input was generated, because ONNX Reshape uses it.
            #     tOp.builtinOptions, opCode = opConvertor.convertReshape(oNode, self.__tflBufferForName)
            #     tOp.opcodeIndex = self.__opCodeIndexForOpType(opCode)
            case _:
                err.error(None, f"Conversion of ONNX Operator '{oNode.opType}' is not yet supported!")

        self.__getOperators().append(tOp)


    def buildInternalTensors(self, oTensors: list[onnxVI.ValueInfo]):
        """ Create 'tensor' tables in the 'tensors' vecotr of the subGraph for oTensors.
            The 'oTensors' do NOT contain data. They should be the inputs and outputs of
            operators in the graph. 
            Designed for the 'value_info' field in ONNX 'Graph'."""
        for oTensor in oTensors:
            if oTensor.type.tensorType is None:
                err.error(err.Code.UNSUPPORTED_ONNX_TYPE,"ONNX: Only type 'tensor_type' is supported for ValueInfo yet!")

            if self.__tensorExists(oTensor.name):
                # Tensor was already created using a different function
                return 

            buffer = self.__buildEmptyBuffer(oTensor.name)
            self.__buildEmptyTensor(oTensor, buffer)


    def buildConstantTensors(self, oTensors: onnxT.Tensors):
        """ Create 'tensor' and 'buffer' tables for the ONNX 'oTensors'.
            The 'oTensors' should have data in them. 
            Designed for the 'initializer' field of the ONNX 'Graph'. """
        for oTensor in oTensors:
            buffer = self.__buildBuffer(oTensor)
            self.__buildConstantTensor(oTensor, buffer)
            

    def buildOutputTensors(self, oOutputs: list[onnxVI.ValueInfo]):
        """ Create 'tensor' tables in the 'tensors' vector of the subGraph for the 'oOutputs'.
            Also create empty buffers in the 'buffers' vector of the model. 
            SHOULD be called before any other tensor building function!
            Designed for the 'output' field of the ONNX 'Graph'. """

        if self.__bufferSize() != 0:
            err.internal("'Builder.buildOutputTensors()' should be called before any other Tensor building function!")

        outputs = tflSG.Outputs()

        for oOutput in oOutputs:
            if oOutput.type.tensorType is None:
                err.error(err.Code.UNSUPPORTED_ONNX_TYPE,"ONNX: Only type 'tensor_type' is supported for Outputs yet!")

            buffer = self.__buildEmptyBuffer(oOutput.name)
            self.__buildEmptyTensor(oOutput, buffer)

            outputs.tmpOutputs.append(self.__tensorForName(oOutput.name))        

        self.__getSubgraph().outputs = outputs


    def buildInputTensors(self, oInputs: list[onnxVI.ValueInfo]):
        """ Create 'tensor' tables in the 'tensors' vector of the subGraph for the 'oInputs'.
            Also create empty buffers in the 'buffers' vector of the model. """

        inputs = tflSG.Inputs()

        for oInput in oInputs:
            if oInput.type.tensorType is None:
                err.error(err.Code.UNSUPPORTED_ONNX_TYPE,"ONNX: Only type 'tensor_type' is supported for Inputs yet!")

            buffer = self.__buildEmptyBuffer(oInput.name)
            self.__buildEmptyTensor(oInput, buffer)

            inputs.tmpInputs.append(self.__tensorForName(oInput.name))        


        self.__getSubgraph().inputs = inputs


    def finish(self) -> tflM.Model:

        # Assign each buffer its index
        for i, buffer in enumerate(self.__getBuffers().vector):
            buffer.tmpIndex = i

        # Assign each tensor its index and its buffer index
        for i, tensor in enumerate(self.__getTensors().vector):
            tensor.tmpIndex = i
            tensor.buffer = tensor.tmpBuffer.tmpIndex

        # Assign 'Outputs' and 'Inputs' their tensor inidces
        outputs = self.__getSubgraph().outputs
        for tensor in outputs.tmpOutputs:
            outputs.append(tensor.tmpIndex)

        inputs = self.__getSubgraph().inputs
        for tensor in inputs.tmpInputs:
            inputs.append(tensor.tmpIndex)

        # Assign each operator its inputs and outputs indices
        for operator in self.__getSubgraph().operators.vector:
            for inputTensor in operator.tmpInputs:
                operator.inputs.append( inputTensor.tmpIndex )

            for outputTensor in operator.tmpOutputs:
                operator.outputs.append( outputTensor.tmpIndex )

        return self.__tflModel




    """ -------------------- Private generic build functions. -------------------- """


    def __buildOperatorCode(self, opType: tflBO.BuiltinOperator):
        """ Add a new OperatorCode for given 'opType' to the 'operator_codes' vector. """
        opCode = tflOC.OperatorCode(opType)

        self.__getOperatorCodes().append(opCode)


    def __buildBuffer(self, oTensor: onnxT.Tensor) -> tflB.Buffer:
        buffer = tflB.Buffer()

        if oTensor.data is None:
            # No data was provided in the tensor
            err.warning(f"ONNX Tensor '{oTensor.name}' should contain data but doesn't! Generating empty buffer!")
            self.__appendNewBuffer(buffer, oTensor.name)
            return


        # Convert the data
        buffer.type = Convertor.convertDataType(oTensor.dataType)
        buffer.data = Convertor.convertTensorData(oTensor.data, oTensor.dims)

        self.__appendNewBuffer(buffer, oTensor.name)

        return buffer


    def __buildConstantTensor(self, oTensor: onnxT.Tensor, buffer: tflB.Buffer):
            shape = Convertor.convertShapeDims(oTensor.dims)
            name = oTensor.name
            type = Convertor.convertDataType(oTensor.dataType)

            tTensor = tflT.Tensor(shape,name,None,type)
            tTensor.tmpBuffer = buffer

            self.__appendNewTensor(tTensor)


    def __buildEmptyTensor(self, oVI :onnxVI.ValueInfo, buffer: tflB.Buffer):
        """ Build a 'Tensor' object from am ONNX ValueInfo object. So the resulting tensor has
            no data, just properties. """
        oTensor = oVI.type.tensorType
        if oTensor is None:
            err.error(err.Code.UNSUPPORTED_ONNX_TYPE,"ONNX: Only type 'tensor_type' is supported yet!")

        shape = Convertor.convertShape(oTensor.shape)
        name = oVI.name
        type = Convertor.convertDataType(oTensor.elemType)

        tensor = tflT.Tensor(shape, name, None, type)
        tensor.tmpBuffer = buffer
        self.__appendNewTensor(tensor)


    def __buildEmptyBuffer(self, name: str) -> tflB.Buffer:
        # TODO comment
        buffer = tflB.Buffer([])

        self.__getBuffers().append(buffer)

        return buffer



    """ -------------------- Private 'quality of life' functions. -------------------- """


    def __opCodeIndexForOpType(self, opType: tflBO.BuiltinOperator):
        """ Return the index to the 'operator_codes' vector in the TFLite model for
            the operator with given 'opType'.
            If corresponding opCode doesn't exist, create new mapping and a new OperatorCode. """
        if opType not in self.__opCodeTypeIndexMap.keys():
            self.__opCodeTypeIndexMap[opType] = self.__operatorCodesSize()
            self.__buildOperatorCode(opType)
        
        return self.__opCodeTypeIndexMap[opType]



    def __tensorExists(self, name: str):
        """ Determine if a tensor with 'name' already exists or not. """
        return name in self.__tensorNameMap.keys()
    

    def __tensorForName(self, name: str) -> tflT.Tensor:
        # TODO comment
        if name not in self.__tensorNameMap.keys():
            err.note(f"Tensor '{name}' is not yet in the tensors. Adding it!") 
            newTensor = tflT.Tensor(tflT.Shape([]),name) # TODO Should be OK (only useless output tensor)
            newTensor.tmpBuffer = self.__buildEmptyBuffer("")
            self.__appendNewTensor(newTensor)

        return self.__tensorNameMap[name]


    def __bufferSize(self):
        """ Return the number of buffers that are currently in the model. """
        return self.__getBuffers().len()



    def __operatorCodesSize(self):
        """ Return the number of buffers that are currently in the model. """
        return len(self.__opCodeTypeIndexMap.keys())


    def __appendNewTensor(self, tTensor: tflT.Tensor):
        # TODO comment

        if tTensor.name in self.__tensorNameMap.keys():
            err.warning(f"Tensor '{tTensor.name}' is already in the tensors!")  
        else:
            self.__tensorNameMap[tTensor.name] = tTensor
            self.__getTensors().append(tTensor)


    def __appendNewBuffer(self, buffer: tflB.Buffer, name: str):
        # TODO comment
        self.__getBuffers().append(buffer)




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

    
    def __getOperators(self) -> tflO.Operators:
        subGraph = self.__getSubgraph()
        if subGraph.operators is None:
            subGraph.operators = tflO.Operators()

        return subGraph.operators

    def __getOperatorCodes(self) -> tflOC.OperatorCodes:
        if self.__tflModel.operatorCodes is None:
            self.__tflModel.operatorCodes = tflOC.OperatorCodes()

        return self.__tflModel.operatorCodes
