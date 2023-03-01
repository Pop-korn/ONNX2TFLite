from typing import List

import src.converter.builder.ModelBuilder as ModelBuilder

import src.generator.model.SubGraphs as tflSG

import src.parser.model.ValueInfo as onnxVI
import src.parser.model.Tensors as onnxT

import src.err as err

class TensorConverter:
    """ This class provides methods to convert ONNX tensors to TFLite and create them 
        using the provided 'ModelBuilder'. """

    __builder: ModelBuilder.ModelBuilder

    def __init__(self, builder: ModelBuilder.ModelBuilder) -> None:
        self.__builder = builder

    def convertInternalTensors(self, oTensors: List[onnxVI.ValueInfo]):
        """ Create 'tensor' tables in the 'tensors' vecotr of the subGraph for oTensors.
            The 'oTensors' do NOT contain data. They should be the inputs and outputs of
            operators in the graph. 
            Designed for the 'value_info' field in ONNX 'Graph'."""
        for oTensor in oTensors:
            if oTensor.type.tensorType is None:
                err.error(err.Code.UNSUPPORTED_ONNX_TYPE,"ONNX: Only type 'tensor_type' is supported for ValueInfo yet!")

            if self.__builder.tensorExists(oTensor.name):
                # Tensor was already created using a different function
                return 

            buffer = self.__builder.buildEmptyBuffer()
            self.__builder.buildEmptyTensor(oTensor, buffer)


    def convertConstantTensors(self, oTensors: onnxT.Tensors):
        """ Create 'tensor' and 'buffer' tables for the ONNX 'oTensors'.
            The 'oTensors' should have data in them. 
            Designed for the 'initializer' field of the ONNX 'Graph'. """
        for oTensor in oTensors:
            buffer = self.__builder.buildBuffer(oTensor)
            self.__builder.buildConstantTensor(oTensor, buffer)
            

    def convertOutputTensors(self, oOutputs: List[onnxVI.ValueInfo]):
        """ Create 'tensor' tables in the 'tensors' vector of the subGraph for the 'oOutputs'.
            Also create empty buffers in the 'buffers' vector of the model. 
            SHOULD be called before any other tensor building function!
            Designed for the 'output' field of the ONNX 'Graph'. """

        if self.__builder.bufferSize() != 0:
            err.internal("'Builder.buildOutputTensors()' should be called before any other Tensor building function!")

        outputs = tflSG.Outputs()

        for oOutput in oOutputs:
            if oOutput.type.tensorType is None:
                err.error(err.Code.UNSUPPORTED_ONNX_TYPE,"ONNX: Only type 'tensor_type' is supported for Outputs yet!")

            buffer = self.__builder.buildEmptyBuffer()
            self.__builder.buildEmptyTensor(oOutput, buffer)

            outputs.tmpOutputs.append(self.__builder.tensorForName(oOutput.name))        

        self.__builder.getSubgraph().outputs = outputs


    def convertInputTensors(self, oInputs: List[onnxVI.ValueInfo]):
        """ Create 'tensor' tables in the 'tensors' vector of the subGraph for the 'oInputs'.
            Also create empty buffers in the 'buffers' vector of the model. """

        inputs = tflSG.Inputs()

        for oInput in oInputs:
            if oInput.type.tensorType is None:
                err.error(err.Code.UNSUPPORTED_ONNX_TYPE,"ONNX: Only type 'tensor_type' is supported for Inputs yet!")

            buffer = self.__builder.buildEmptyBuffer()
            self.__builder.buildEmptyTensor(oInput, buffer)

            inputs.tmpInputs.append(self.__builder.tensorForName(oInput.name))        


        self.__builder.getSubgraph().inputs = inputs
