import src.generator.builtin.Add as tflAdd
import src.generator.model.Operators as tflO
import src.generator.meta.meta as meta
import src.err as err

def convert(tOp: tflO.Operator) -> meta.BuiltinOptions:
    
    if len(tOp.tmpInputs) == 2:
        """ Can use 'Add' operator to represent the 'Sum'. """

        return tflAdd.Add()

    else:
        err.error(err.Code.UNSUPPORTED_OPERATOR,
                  f"CvtSum: operator has '{len(tOp.tmpInputs)}' inputs!",
                  "Conversion not yet implemented!")
