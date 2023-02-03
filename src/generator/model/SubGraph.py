import flatbuffers as fb

import tflite.SubGraph as sg
import tflite.Model as Model

import generator.model.Tensor as Tensor
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

    def __init__(self, inputs: Inputs, outputs: Outputs, tensors: Tensor.Tensors):
        self.inputs = inputs
        self.outputs = outputs
        self.tensors = tensors

    def genTFLite(self, builder: fb.Builder):
        inputsTFLite = self.inputs.genTFLite(builder)
        outputsTFLite = self.outputs.genTFLite(builder)
        tensorsTFLite = self.tensors.genTFLite(builder)

        sg.Start(builder)

        sg.AddInputs(builder,inputsTFLite)
        sg.AddOutputs(builder, outputsTFLite)
        sg.AddTensors(builder,tensorsTFLite)

        return sg.End(builder)

def genSubGraphs(builder: fb.Builder, subGraphs: list[SubGraph]):
    tfliteSubGraphs = []

    for subGraph in subGraphs:
        tfliteSubGraphs.append(subGraph.genTFLite(builder))

    Model.StartSubgraphsVector(builder, len(subGraphs))

    for tfliteSubGraph in tfliteSubGraphs:
        builder.PrependSOffsetTRelative(tfliteSubGraph)

    return builder.EndVector()
