from typing import List
import flatbuffers as fb

import src.generator.meta.meta as meta

import lib.tflite.OperatorCode as oc
import lib.tflite.Model as Model
import lib.tflite.BuiltinOperator as bo

class OperatorCode(meta.TFLiteObject):
    """ Represents an OperatorCode object, used in the vector 'operator_codes' in the model.
    """

    builtinCode: bo.BuiltinOperator
    version: int
    # TODO customCode

    def __init__(self,builtinCode: bo.BuiltinOperator,
                version: int=1):
        """
        Args:
            builtinCode (BuiltinOperator): operator code from the 'BuiltinOperator' enum
            version (int, optional): operator version. Defaults to 1.
        """
        self.version = version
        self.builtinCode = builtinCode

    def genTFLite(self,builder: fb.builder):
        """ Generate TFLite representation for this OperatorCode """
        oc.Start(builder)
        oc.AddDeprecatedBuiltinCode(builder,self.builtinCode)
        oc.AddBuiltinCode(builder,self.builtinCode)
        oc.AddVersion(builder,self.version)
        
        return oc.End(builder)

class OperatorCodes(meta.TFLiteVector):
    def __init__(self, operatorCodes: List[OperatorCode] = []) -> None:
        super().__init__(operatorCodes,Model.StartOperatorCodesVector)
    