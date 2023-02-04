import flatbuffers as fb

import tflite.SubGraph as sg
import tflite.Model as Model

import generator.model.Tensors as Tensors
import generator.model.Operators as Operators
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
    tensors: Tensors.Tensors
    operators: Operators.Operators

    def __init__(self, inputs: Inputs, outputs: Outputs, tensors: Tensors.Tensors, operators: Operators.Operators):
        self.inputs = inputs
        self.outputs = outputs
        self.tensors = tensors
        self.operators = operators

    def genTFLite(self, builder: fb.Builder):
        tflInputs = self.inputs.genTFLite(builder)
        tflOutputs = self.outputs.genTFLite(builder)
        tflTensors = self.tensors.genTFLite(builder)
        tflOperators = self.operators.genTFLite(builder)

        sg.Start(builder)

        sg.AddInputs(builder, tflInputs)
        sg.AddOutputs(builder, tflOutputs)
        sg.AddTensors(builder, tflTensors)
        sg.AddOperators(builder, tflOperators)

        return sg.End(builder)

def genSubGraphs(builder: fb.Builder, subGraphs: list[SubGraph]):
    tflSubGraphs = []

    for subGraph in subGraphs:
        tflSubGraphs.append(subGraph.genTFLite(builder))

    Model.StartSubgraphsVector(builder, len(subGraphs))

    for tflSubGraph in tflSubGraphs:
        builder.PrependSOffsetTRelative(tflSubGraph)

    return builder.EndVector()
