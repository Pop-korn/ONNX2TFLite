import src.parser.builtin.Constant as onnxConstant

import src.converter.builder.ModelBuilder as ModelBuilder

def convert(oC: onnxConstant.Constant, 
            modelBuilder: ModelBuilder.ModelBuilder) -> None:
    
    modelBuilder.createTensorForData(oC.value.data, oC.value.name)

    return None

