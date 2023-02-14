import parser.model.Model as m

import parser.builtin.Conv as c

model = m.Model("data/onnx/bvlcalexnet-12.onnx")

def handleConvOp(conv: c.Conv):
    print(conv.pads)
    print(conv.kernelShape)
    print(conv.strides)
    

if model.graph.nodes[0].opType == "Conv":
    handleConvOp(model.graph.nodes[0].attributes)

