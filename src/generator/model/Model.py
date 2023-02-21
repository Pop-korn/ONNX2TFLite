import flatbuffers as fb

import src.generator.model.OperatorCodes as oc
import src.generator.model.SubGraphs as sg
import src.generator.model.Buffers as b

import src.generator.meta.meta as meta

import src.err as err

import lib.tflite.Model as m

class Model(meta.TFLiteObject):
    version: int
    description: str
    operatorCodes: oc.OperatorCodes
    subGraphs: sg.SubGraphs
    buffers: b.Buffers
    # TODO signatureDefs
    # TODO metadata
    # TODO metadataBuffer

    __fileIdentifier = "TFL3" # file_identifier from the used TFLite schema

    @classmethod
    def __genFileIdentifier(cls):
        """ Generate byte-like object representing the TFLite format """
        return cls.__fileIdentifier.encode("ascii")

    def __init__(self, version: int=1,
                description: str=None,
                buffers: b.Buffers=None,
                operatorCodes: oc.OperatorCodes=None,
                subGraphs: sg.SubGraphs=None) -> None:
        self.version = version
        self.description = description
        self.operatorCodes = operatorCodes
        self.subGraphs = subGraphs
        self.buffers = buffers

    def genTFLite(self, builder: fb.Builder):
        if self.description is not None:
            err.expectType(self.description, str, "Model.description")
            tflDescription = builder.CreateString(self.description)
        
        err.expectType(self.operatorCodes, oc.OperatorCodes, "Model.operatorCodes")
        if self.operatorCodes is not None:
            tflOperatorCodes = self.operatorCodes.genTFLite(builder)
        
        err.expectType(self.subGraphs, sg.SubGraphs, "Model.subGraphs")
        if self.subGraphs is not None:
            tflSubGraphs = self.subGraphs.genTFLite(builder)
        
        err.expectType(self.buffers, b.Buffers, "Model.buffers")
        if self.buffers is not None:
            tflBuffers = self.buffers.genTFLite(builder)

        m.Start(builder)
            
        m.AddVersion(builder,self.version)

        if self.description is not None:
            m.AddDescription(builder,tflDescription)
        if self.operatorCodes is not None:
            m.AddOperatorCodes(builder,tflOperatorCodes)
        if self.subGraphs is not None:
            m.AddSubgraphs(builder,tflSubGraphs)
        if self.buffers is not None:
            m.AddBuffers(builder,tflBuffers)

        builder.Finish(m.End(builder),Model.__genFileIdentifier())
