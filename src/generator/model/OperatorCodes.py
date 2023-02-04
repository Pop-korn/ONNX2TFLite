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
        """ Generate TFLite representation for this OperatorCode """
        oc.Start(builder)
        oc.AddDeprecatedBuiltinCode(builder,self.builtinCode)
        oc.AddBuiltinCode(builder,self.builtinCode)
        oc.AddVersion(builder,self.version)
        
        return oc.End(builder)

class OperatorCodes:
    operatorCodes: list[OperatorCode]

    def __init__(self, operatorCodes: list[OperatorCode] = []) -> None:
        self.operatorCodes = operatorCodes

    def append(self, operatorCode: OperatorCode):
        self.operatorCodes.append(operatorCode)

    def get(self, index: int):
        return self.operatorCodes[index]

    def genTFLite(self, builder):
        """Generate TFLite representation for all OperatorCodes in 'operatorCodes' """
        tflOperatorCodes = [opCode.genTFLite(builder) for opCode in self.operatorCodes]

        Model.StartOperatorCodesVector(builder, len(self.operatorCodes))

        for tflOperatorCode in tflOperatorCodes:
            builder.PrependSOffsetTRelative(tflOperatorCode) # TODO check

        return builder.EndVector()


    