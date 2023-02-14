import onnx.onnx.onnx_ml_pb2 as onnx

import err

import parser.model.OperatorSetId as osi
import parser.model.Graph as g

class Model():
    # Wrapped descriptor
    __model: onnx.ModelProto

    # Model Attributes
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
        self.__model = onnx.ModelProto()
        self.__loadModel(srcFile)
        self.__initAttributes()

    def __initAttributes(self):
        """ Initialize object attributes from the '__model' descriptor. """
        self.irVersion = self.__model.ir_version
        self.opsetImport = osi.OperatorSetIds(self.__model.opset_import)
        self.producerName = self.__model.producer_name
        self.producerVersion = self.__model.producer_version
        self.domain = self.__model.domain
        self.modelVersion = self.__model.model_version
        self.docString = self.__model.doc_string
        self.graph = g.Graph(self.__model.graph)

    def __loadModel(self, srcFile):
        """ Load data from onnx model stored in give file. """
        try:
            raw_data = None
            with open(srcFile,"rb") as f:
                raw_data = f.read()
            try:
                self.__model.ParseFromString(raw_data)
            except:
                err.eprint(err.Code.INPUT_FILE_ERR, f"Cannot parse file '{srcFile}'")
        except:
            err.eprint(err.Code.INPUT_FILE_ERR, f"Cannot read from file '{srcFile}'")

