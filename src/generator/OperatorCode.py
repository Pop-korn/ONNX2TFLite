import flatbuffers as fb

import tflite.OperatorCode as oc
import tflite.Model as Model
import tflite.BuiltinOperator as bo

class OperatorCode:
    """ Represents an OperatorCode object, used in the vector 'operator_codes' in the model.
    """

    builtinCode: bo.BuiltinOperator
    version: int
    # TODO customCode

    def __init__(self,builtinCode: bo.BuiltinOperator,version: int=1):
        """

        Args:
            builtinCode (BuiltinOperator): operator code from the 'BuiltinOperator' enum
            version (int, optional): operator version. Defaults to 1.
        """
        self.version = version
        self.builtinCode = builtinCode

    def genTFLite(self,builder: fb.builder):
        """Generate TFLite representation for this OperatorCode

        Args:
            builder (fb.builder):

        Returns:
            int: TFLite representation of the OperatorCode
        """
        oc.Start(builder)
        oc.AddDeprecatedBuiltinCode(builder,self.builtinCode)
        oc.AddBuiltinCode(builder,self.builtinCode)
        oc.AddVersion(builder,self.version)
        
        return oc.End(builder)

def genOperatorCodes(builder: fb.builder,operatorCodes: list[OperatorCode]):
    """Generate TFLite representation for all OperatorCodes in 'operatorCodes' 

    Args:
        builder (fb.builder):
        operatorCodes (list[OperatorCode]): List of 'OperatorCode' objects to generate TFLite for

    Returns:
        All: TFLite struct 'operator_codes'
    """
    tfliteOpCodes = []

    for opCode in operatorCodes:
        tfliteOpCodes.append(opCode.genTFLite(builder))

    Model.StartOperatorCodesVector(builder,1)

    for tfliteOpCode in tfliteOpCodes:
        builder.PrependSOffsetTRelative(tfliteOpCode)

    return builder.EndVector()