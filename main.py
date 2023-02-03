import flatbuffers as fb
import tflite.Model as Model
import tflite.BuiltinOperator as BuiltinOperator

import generator.OperatorCode as OperatorCode
import generator.SubGraph as SubGraph

def gen_Model(builder: fb.Builder):
    desc = builder.CreateString("Popis modelu")

    # Operator Codes
    operatorCodes = [OperatorCode.OperatorCode(BuiltinOperator.BuiltinOperator.CONV_2D)]
    opCodesFB = OperatorCode.genOperatorCodes(builder,operatorCodes)

    # SubGraphs (only 1 so far)
    subgraph = SubGraph.genSubgraph(builder)
    Model.StartSubgraphsVector(builder,1)
    builder.PrependSOffsetTRelative(subgraph)
    subgraphs = builder.EndVector()


    # Create Model
    Model.Start(builder)

    Model.AddVersion(builder,5)
    Model.AddDescription(builder,desc)
    Model.AddOperatorCodes(builder,opCodesFB)
    Model.AddSubgraphs(builder,subgraphs)

    builder.Finish(Model.End(builder))

builder = fb.Builder(1024)

gen_Model(builder)

# gen_conv2d(builder)



buffer = builder.Output()

with open("test/out.tflite","wb") as f:
    f.write(buffer)