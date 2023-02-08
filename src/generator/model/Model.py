import flatbuffers as fb

import generator.model.OperatorCodes as oc
import generator.model.SubGraphs as sg
import generator.model.Buffers as b

import generator.meta.meta as meta

import tflite.Model as m

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
        tflDescription = builder.CreateString(self.description)
        tflOperatorCodes = self.operatorCodes.genTFLite(builder)
        tflSubGraphs = self.subGraphs.genTFLite(builder)
        tflBuffers = self.buffers.genTFLite(builder)

        m.Start(builder)

        m.AddVersion(builder,self.version)
        m.AddDescription(builder,tflDescription)
        m.AddOperatorCodes(builder,tflOperatorCodes)
        m.AddSubgraphs(builder,tflSubGraphs)
        m.AddBuffers(builder,tflBuffers)

        return m.End(builder)

    def Finish(self, builder: fb.Builder, tflModel: int):
        """ Finish generating TFLite for this model. Model must have been built with
            the provided 'builder'. Resulting output will be stored in 'builder.Output()'.

        Args:
            builder (fb.Builder): flatbuffers.Builder object, that was used to build the flatbuffer
            tflModel (int): offset of the built model (result of 'Model.genTFLite()')
        """
        builder.Finish(tflModel,Model.__genFileIdentifier())
