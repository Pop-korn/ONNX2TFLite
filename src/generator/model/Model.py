import flatbuffers as fb

import generator.model.OperatorCodes as oc
import generator.model.SubGraphs as sg

import generator.meta.meta as meta

import tflite.Model as m

class Model(meta.TFLiteObject):
    version: int
    description: str
    operatorCodes: oc.OperatorCodes
    subGraphs: sg.SubGraphs

    def __init__(self, version: int = 1, description: str = None
    , operatorCodes: oc.OperatorCodes = None, subGraphs: sg.SubGraphs = None) -> None:
        self.version = version
        self.description = description
        self.operatorCodes = operatorCodes
        self.subGraphs = subGraphs

    def genTFLite(self, builder: fb.Builder):
        tflDescription = builder.CreateString(self.description)
        tflOperatorCodes = self.operatorCodes.genTFLite(builder)
        tflSubGraphs = self.subGraphs.genTFLite(builder)

        m.Start(builder)

        m.AddVersion(builder,self.version)
        m.AddDescription(builder,tflDescription)
        m.AddOperatorCodes(builder,tflOperatorCodes)
        m.AddSubgraphs(builder,tflSubGraphs)

        builder.Finish(m.End(builder))
