import flatbuffers as fb
import tflite.Model as Model
import tflite.SubGraph as SubGraph

import generator.OperatorCode as OperatorCode

def gen_Subgraph(builder: fb.Builder):

    SubGraph.Start(builder)

    return SubGraph.End(builder)


def gen_Subgraphs(builder: fb.Builder):
    subgraphs = []

    subgraphs.append(gen_Subgraph(builder))

    return subgraphs

def gen_Model(builder: fb.Builder):
    desc = builder.CreateString("Popis modelu")

    # Operator Codes
    opCodes = OperatorCode.genOperatorCodes(builder) # TODO add parameters

    # SubGraphs
    subgraphs = gen_Subgraphs(builder)
    Model.StartSubgraphsVector(builder,1)
    for subgraph in subgraphs:
        builder.PrependSOffsetTRelative(subgraph)
    subgraphs = builder.EndVector()

    Model.Start(builder)

    Model.AddVersion(builder,5)
    Model.AddDescription(builder,desc)
    Model.AddOperatorCodes(builder,opCodes)
    Model.AddSubgraphs(builder,subgraphs)

    # Model.AddOperatorCodes(builder,)

    builder.Finish(Model.End(builder))



builder = fb.Builder(1024)

gen_Model(builder)

# gen_conv2d(builder)



buffer = builder.Output()

with open("test/out.tflite","wb") as f:
    f.write(buffer)