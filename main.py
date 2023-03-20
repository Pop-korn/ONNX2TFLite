import src.converter.convert as convert

""" Temporary. Handle which models to convert. """
# convert.convertModel("data/onnx/bvlcalexnet-12.onnx", "test/alexnet.tflite")
convert.convertModel("data/onnx/tinyyolov2-8.onnx", "test/tinyyolo.tflite")

exit()

from src.generator.model import Model, SubGraphs, Buffers, Operators, OperatorCodes, Tensors
from src.generator.builtin import Conv2D
import flatbuffers as fb
import numpy as np

builder = fb.Builder()

m = Model.Model()

m.subGraphs = SubGraphs.SubGraphs([SubGraphs.SubGraph(
    operators= Operators.Operators([
        Operators.Operator(
        )
    ]),
    tensors= Tensors.Tensors([
        Tensors.Tensor(
            Tensors.Shape([]),
            name=""
        )
    ])
)])

# m.operatorCodes = OperatorCodes.OperatorCodes([
#     OperatorCodes.OperatorCode(3)
# ])

# m.buffers = Buffers.Buffers([Buffers.Buffer( 
#     np.ones([10],np.int32)
# )])






#-----------------------------------------------

m.genTFLite(builder)

buffer = builder.Output()

with open("test/test.tflite", "wb") as f:
    f.write(buffer)







exit()
import src.converter.convert as convert

""" Temporary. Handle which models to convert. """
# convert.convertModel("data/onnx/bvlcalexnet-12.onnx", "test/alexnet.tflite")
convert.convertModel("data/onnx/tinyyolov2-8.onnx", "test/tinyyolo.tflite")

