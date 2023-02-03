import flatbuffers as fb
import tflite.SubGraph as SubGraph

def genSubgraph(builder: fb.Builder):

    SubGraph.Start(builder)


    return SubGraph.End(builder)

# def genSubgraphs(builder: fb.Builder):
    subgraphs = []

    subgraphs.append(genSubgraph(builder))

    return subgraphs