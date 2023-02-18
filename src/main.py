import parser.model.Model as m

import parser.builtin.Conv as Conv
import parser.builtin.Gemm as Gemm
import parser.builtin.LRN as LRN
import parser.builtin.MaxPool as MaxPool
import parser.builtin.Softmax as Softmax

model = m.Model("data/onnx/bvlcalexnet-12.onnx")

print(model.docString,model.domain,model.irVersion,model.modelVersion,model.producerName,model.producerVersion)
print(model.opsetImport[0].domain, model.opsetImport[0].version)

input = model.graph.inputs[0]

print(input.name,input.docString)

if not input.type.tensorType is None:
    for dim in input.type.tensorType.shape.dims:
        print(dim.value)


# Operator Tests...

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

def handleMaxPool(mp: MaxPool.MaxPool):
    print("MaxPool")
    print(mp.autoPad)
    print(mp.ceilMode)
    print(mp.dilations)
    print(mp.kernelShape)
    print(mp.pads)
    print(mp.storageOrder)
    print(mp.strides)
    print("")

def handleGemmOp(gemm: Gemm.Gemm):
    print("Gemm")
    print(gemm.alpha)
    print(gemm.beta)
    print(gemm.transA)
    print(gemm.transB)
    print("")

def handleSoftmaxOp(s: Softmax.Softmax):
    print("Softmax")
    print(s.axis)
    print("")


# for node in model.graph.nodes:
#     if node.opType == "Conv":
#         handleConvOp(node.attributes)
#     elif node.opType == "Softmax":
#         handleSoftmaxOp(node.attributes)
#     elif node.opType == "Gemm":
#         handleGemmOp(node.attributes)
#     elif node.opType == "LRN":
#         handleLRN(node.attributes)
#     elif node.opType == "MaxPool":
#         handleMaxPool(node.attributes)

