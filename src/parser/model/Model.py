import onnx.onnx.onnx_ml_pb2 as onnx

import err

import parser.meta.meta as meta

import parser.model.OperatorSetId as osi
import parser.model.Graph as g

class Model(meta.ONNXObject):
    irVersion: int
    opsetImport: osi.OperatorSetIds
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
        self.opsetImport = osi.OperatorSetIds(self._descriptor.opset_import)
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
                err.eprint(err.Code.INPUT_FILE_ERR, f"Cannot parse file '{srcFile}'")
        except:
            err.eprint(err.Code.INPUT_FILE_ERR, f"Cannot read from file '{srcFile}'")

