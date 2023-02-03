import flatbuffers as fb
import tflite.SubGraph as sg
import tflite.Model as Model


class Inputs:
    inputs: list[int] = None

    def __init__(self, inputs: list[int]):
        self.inputs = inputs

    def genTFLite(self, builder: fb.Builder):
        sg.StartInputsVector(builder, len(self.inputs))
        
        for input in self.inputs:
            builder.PrependInt32(input)

        return builder.EndVector()

class Outputs:
    outputs: list[int] = None

    def __init__(self, outputs: list[int]):
        self.outputs = outputs

    def genTFLite(self, builder: fb.Builder):
        sg.StartOutputsVector(builder, len(self.outputs))
        
        for output in self.outputs:
            builder.PrependInt32(output)

        return builder.EndVector() 

class SubGraph:
    inputs: Inputs = None
    outputs: Outputs = None

    def __init__(self, inputs: Inputs, outputs: Outputs):
        self.inputs = inputs
        self.outputs = outputs

    def genTFLite(self, builder: fb.Builder):
        inputsTFLite = self.inputs.genTFLite(builder)
        outputsTFLite = self.outputs.genTFLite(builder)

        sg.Start(builder)

        sg.AddInputs(builder,inputsTFLite)
        sg.AddOutputs(builder, outputsTFLite)

        return sg.End(builder)

def genSubGraphs(builder: fb.Builder, subGraphs: list[SubGraph]):
    tfliteSubGraphs = []

    for subGraph in subGraphs:
        tfliteSubGraphs.append(subGraph.genTFLite(builder))

    Model.StartSubgraphsVector(builder, len(subGraphs))

    for tfliteSubGraph in tfliteSubGraphs:
        builder.PrependSOffsetTRelative(tfliteSubGraph)

    return builder.EndVector()
