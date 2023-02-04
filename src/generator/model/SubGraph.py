import flatbuffers as fb

import tflite.SubGraph as sg
import tflite.Model as Model

import generator.model.Tensor as Tensor
import generator.model.Operator as Operator
import generator.meta.meta as meta

""" Classes representing the 'SubGraph' structure and its parameters """

class Inputs(meta.IntVector):
    def __init__(self, inputs: list[int]):
        super().__init__(inputs,sg.StartInputsVector)

class Outputs(meta.IntVector):
    def __init__(self, outputs: list[int]):
        super().__init__(outputs,sg.StartOutputsVector)


class SubGraph:
    inputs: Inputs
    outputs: Outputs
    tensors: Tensor.Tensors
    operators: Operator.Operators

    def __init__(self, inputs: Inputs, outputs: Outputs, tensors: Tensor.Tensors, operators: Operator.Operators):
        self.inputs = inputs
        self.outputs = outputs
        self.tensors = tensors
        self.operators = operators

    def genTFLite(self, builder: fb.Builder):
        inputsTFLite = self.inputs.genTFLite(builder)
        outputsTFLite = self.outputs.genTFLite(builder)
        tensorsTFLite = self.tensors.genTFLite(builder)
        operatorsTFLite = self.operators.genTFLite(builder)

        sg.Start(builder)

        sg.AddInputs(builder, inputsTFLite)
        sg.AddOutputs(builder, outputsTFLite)
        sg.AddTensors(builder, tensorsTFLite)
        sg.AddOperators(builder, operatorsTFLite)

        return sg.End(builder)

def genSubGraphs(builder: fb.Builder, subGraphs: list[SubGraph]):
    tfliteSubGraphs = []

    for subGraph in subGraphs:
        tfliteSubGraphs.append(subGraph.genTFLite(builder))

    Model.StartSubgraphsVector(builder, len(subGraphs))

    for tfliteSubGraph in tfliteSubGraphs:
        builder.PrependSOffsetTRelative(tfliteSubGraph)

    return builder.EndVector()
