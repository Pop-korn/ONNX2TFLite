import flatbuffers as fb

import lib.tflite.SubGraph as sg
import lib.tflite.Model as Model

import src.generator.model.Tensors as Tensors
import src.generator.model.Operators as Operators
import src.generator.meta.meta as meta

import src.err as err

""" Classes representing the 'SubGraph' st  ructure and its parameters """

class Inputs(meta.IntVector):

    """ List of 'Tensor' objects. Easier to use while converting. """
    tmpInputs: list[Tensors.Tensor]

    def __init__(self, inputs: list[int] = None):
        """ 'inputs' is a list of indices into the 'tensors' vector. """
        super().__init__(inputs,sg.StartInputsVector)
        self.tmpInputs = []


class Outputs(meta.IntVector):

    """ List of 'Tensor' objects. Easier to use while converting. """
    tmpOutputs: list[Tensors.Tensor]

    def __init__(self, outputs: list[int] = None):
        """ 'outputs' is a list of indices into the 'tensors' vector. """
        super().__init__(outputs,sg.StartOutputsVector)
        self.tmpOutputs = []


class SubGraph(meta.TFLiteObject):
    inputs: Inputs
    outputs: Outputs
    tensors: Tensors.Tensors
    operators: Operators.Operators

    def __init__(self, inputs: Inputs=None, outputs: Outputs=None,
                tensors: Tensors.Tensors=None,
                operators: Operators.Operators=None):
        self.inputs = inputs
        self.outputs = outputs
        self.tensors = tensors
        self.operators = operators

    def genTFLite(self, builder: fb.Builder):
        err.expectType(self.inputs, Inputs, "SubGraph.inputs")
        if self.inputs is not None:
            tflInputs = self.inputs.genTFLite(builder)

        err.expectType(self.outputs, Outputs, "SubGraph.outputs")
        if self.outputs is not None:
            tflOutputs = self.outputs.genTFLite(builder)

        err.expectType(self.tensors, Tensors.Tensors, "SubGraph.tensors")
        if self.tensors is not None:
            tflTensors = self.tensors.genTFLite(builder)

        err.expectType(self.operators, Operators.Operators, "SubGraph.operators")
        if self.operators is not None:
            tflOperators = self.operators.genTFLite(builder)

        sg.Start(builder)
        
        if self.inputs is not None:
            sg.AddInputs(builder, tflInputs)
        if self.outputs is not None:
            sg.AddOutputs(builder, tflOutputs)
        if self.tensors is not None:
            sg.AddTensors(builder, tflTensors)
        if self.operators is not None:
            sg.AddOperators(builder, tflOperators)

        return sg.End(builder)

class SubGraphs(meta.TFLiteVector):
    def __init__(self, subGraphs: list[SubGraph] = []) -> None:
        super().__init__(subGraphs,Model.StartSubgraphsVector)
