import parser.model.Model as m

import parser.builtin.Conv as c
import parser.builtin.Gemm as g
import parser.builtin.LRN as lrn
import parser.builtin.MaxPool as mp

model = m.Model("data/onnx/bvlcalexnet-12.onnx")

def handleConvOp(conv: c.Conv):
    print("Conv")
    print(conv.pads)
    print(conv.kernelShape)
    print(conv.strides)
    print(conv.group)
    print("")

def handleLRN(lrn: lrn.LRN):
    print("LRN")
    print(lrn.alpha)
    print(lrn.beta)
    print(lrn.bias)
    print(lrn.size)
    print("")

def handleMaxPool(mp: mp.MaxPool):
    print("MaxPool")
    print(mp.autoPad)
    print(mp.ceilMode)
    print(mp.dilations)
    print(mp.kernelShape)
    print(mp.pads)
    print(mp.storageOrder)
    print(mp.strides)
    print("")

def handleGemmOp(gemm: g.Gemm):
    print("Gemm")
    print(gemm.alpha)
    print(gemm.beta)
    print(gemm.transA)
    print(gemm.transB)
    print("")


for node in model.graph.nodes:
    if node.opType == "Conv":
        handleConvOp(node.attributes)
    elif node.opType == "Gemm":
        handleGemmOp(node.attributes)
    elif node.opType == "LRN":
        handleLRN(node.attributes)
    elif node.opType == "MaxPool":
        handleMaxPool(node.attributes)

