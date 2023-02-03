import flatbuffers as fb
import tflite.SubGraph as sg
import tflite.Model as Model


class Inputs:
    def __init__(self, inputs: list[int]):
        self.inputs = inputs

    def genTFLite(self, builder: fb.Builder):
        sg.StartInputsVector(builder, len(self.inputs))
        
        for input in self.inputs:
            builder.PrependInt32(input)

        return builder.EndVector()

class SubGraph:
    def __init__(self, inputs: Inputs):
        self.inputs = inputs

    def genTFLite(self, builder: fb.Builder):
        inputsTFLite = self.inputs.genTFLite(builder)

        sg.Start(builder)

        sg.AddInputs(builder,inputsTFLite)

        return sg.End(builder)

def genSubGraphs(builder: fb.Builder, subGraphs: list[SubGraph]):
    tfliteSubGraphs = []

    for subGraph in subGraphs:
        tfliteSubGraphs.append(subGraph.genTFLite(builder))

    Model.StartSubgraphsVector(builder, len(subGraphs))

    for tfliteSubGraph in tfliteSubGraphs:
        builder.PrependSOffsetTRelative(tfliteSubGraph)

    return builder.EndVector()
