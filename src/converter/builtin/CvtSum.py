from src.generator.builtin import(
    Add as tflAdd, AddN as tflAddN    
)
import src.generator.model.Operators as tflO
import src.generator.meta.meta as meta
import src.err as err

def convert(tOp: tflO.Operator) -> meta.BuiltinOptions:

    match len(tOp.tmpInputs):
        case 1:
            """ Operator can be skipped """
            # TODO
            err.error(err.Code.UNSUPPORTED_OPERATOR,
                      "CvtSum: 'Sum' has 1 input. Implement conversion!")

        case 2:
            """ Can use 'Add' operator to represent the 'Sum'. This is better than
                just using 'AddN', because of inference speed and possible 
                activation function fusing."""
            return tflAdd.Add()
        
        case _:
            """ 'Sum' has many inputs. Convert to 'AddN'. """
            return tflAddN.AddN()
