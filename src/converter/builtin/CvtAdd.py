import src.generator.builtin.Add as tflAdd
import src.generator.model.Operators as tflO
import src.converter.builder.ModelBuilder as ModelBuilder


def convert(tOp: tflO.Operator, 
            builder: ModelBuilder.ModelBuilder) -> tflAdd.Add:
    for t in tOp.tmpInputs:
        if builder.tensorHasData(t):
            if all( val == 0  for val in t.tmpBuffer.data):
                # The operator is just adding 0 to everything. Skip it
                return None 

    return tflAdd.Add()
