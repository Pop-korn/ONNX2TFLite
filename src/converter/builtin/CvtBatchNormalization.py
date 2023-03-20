import src.parser.builtin.BatchNormalization as onnxBN

import src.generator.model.Operators as tflO

import src.converter.builder.ModelBuilder as ModelBuilder

def convert(oBN: onnxBN.BatchNormalization, 
            tOp: tflO.Operator,
            modelBuilder: ModelBuilder):
    pass