import parser.model.Model as m

import parser.builtin.Conv as Conv
import parser.builtin.LRN as LRN

model = m.Model("data/onnx/bvlcalexnet-12.onnx")

def handleConvOp(conv: Conv.Conv):
    print("Conv")
    print(conv.pads)
    print(conv.kernelShape)
    print(conv.strides)
    print(conv.group)
    print("")

def handleLRN(lrn: LRN.LRN):
    print("LRN")
    print(lrn.alpha)
    print(lrn.beta)
    print(lrn.bias)
    print(lrn.size)
    print("")

for node in model.graph.nodes:
    if node.opType == "Conv":
        handleConvOp(node.attributes)
    elif node.opType == "LRN":
        handleLRN(node.attributes)

