import flatbuffers as fb
import tflite.OperatorCode as oc
import tflite.BuiltinOperator as BuiltinOperator
import tflite.Model as Model

def genOperatorCode(builder: fb.builder,description: str,code: int,version: int=1):
    """Generate an OperatorCode object

    Args:
        builder (fb.builder):
        description (str): OperatorCode description
        code (int): BuiltinOperator code
        version (int, optional): OperatorCode version. Defaults to 1.

    Returns:
        Any: OperatorCode object
    """
    desc = builder.CreateString(description)

    oc.Start(builder)
    oc.AddDeprecatedBuiltinCode(builder,code)
    oc.AddBuiltinCode(builder,code)
    oc.AddVersion(builder,version)
    
    return oc.End(builder)

def genOperatorCodes(builder: fb.builder):
    opCodes = []

    opCodes.append(genOperatorCode(builder,"Konvolucia",BuiltinOperator.BuiltinOperator.CONV_2D))
    
    Model.StartOperatorCodesVector(builder,1)
    for opCode in opCodes:
        builder.PrependSOffsetTRelative(opCode)

    return builder.EndVector()