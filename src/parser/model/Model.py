"""
    Model

Representation of an ONNX 'Model' object.
Initialized from a protobuf descriptor.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

import lib.onnx.onnx.onnx_ml_pb2 as onnx

import src.err as err

import src.parser.meta.meta as meta

import src.parser.model.OperatorSetIds as osi
import src.parser.model.Graph as g

class Model(meta.ONNXObject):
    irVersion: int
    opsetImports: osi.OperatorSetIds
    producerName: str
    producerVersion: str
    domain: str
    modelVersion: int
    docString: str
    graph: g.Graph
    # TODO metadataProps
    # TODO trainingInfo
    # TODO functions


    def __init__(self, srcFile: str) -> None:
        super().__init__(onnx.ModelProto())
        self.__loadModel(srcFile)
        self.__initAttributes()

    def __initAttributes(self):
        """ Initialize object attributes from the '_descriptor' attribute of the parent object. """
        self.irVersion = self._descriptor.ir_version
        self.opsetImports = osi.OperatorSetIds(self._descriptor.opset_import)
        self.producerName = self._descriptor.producer_name
        self.producerVersion = self._descriptor.producer_version
        self.domain = self._descriptor.domain
        self.modelVersion = self._descriptor.model_version
        self.docString = self._descriptor.doc_string
        self.graph = g.Graph(self._descriptor.graph)

    def __loadModel(self, srcFile):
        """ Load data from onnx model stored in give file. """
        try:
            raw_data = None
            with open(srcFile,"rb") as f:
                raw_data = f.read()
            try:
                self._descriptor.ParseFromString(raw_data)
            except:
                err.error(err.Code.INPUT_FILE_ERR, f"Cannot parse file '{srcFile}'")
        except:
            err.error(err.Code.INPUT_FILE_ERR, f"Cannot read from file '{srcFile}'")

