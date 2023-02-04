import flatbuffers as fb
import tflite.Model as Model
import tflite.BuiltinOperator as BuiltinOperator
import tflite.TensorType as TensorType

import generator.model.OperatorCode as OperatorCode
import generator.model.SubGraph as SubGraph
import generator.model.Tensor as Tensor
import generator.model.Quantization as Quant
import generator.model.Operator as Operator

def gen_Model(builder: fb.Builder):
    desc = builder.CreateString("Model Description")

    # Operator Codes
    operatorCodes = [OperatorCode.OperatorCode(BuiltinOperator.BuiltinOperator.CONV_2D)]
    opCodesTFLite = OperatorCode.genOperatorCodes(builder,operatorCodes)

    # SubGraphs
    quantization = Quant.Quantisation(Quant.Min([0.0]), Quant.Max([0.996094])
                    , Quant.Scale([0.003906]), Quant.ZeroPoint([0]), 0)
    tensors = Tensor.Tensors([Tensor.Tensor(quantization, Tensor.Shape([1,10]), 
    "CifarNet/Predictions/Reshape_1",17,TensorType.TensorType.UINT8)])

    inputs = SubGraph.Inputs([0])
    outputs = SubGraph.Outputs([2,3,5])

    operators = Operator.Operators([Operator.Operator(Operator.Inputs([16,3,1]),Operator.Outputs([2]),
        Operator.MutatingVariableInputs([]))])

    subGraphs = [SubGraph.SubGraph(inputs, outputs, tensors,operators)]
    subGraphsTFLite = SubGraph.genSubGraphs(builder,subGraphs)

    # Create Model
    Model.Start(builder)

    Model.AddVersion(builder,5)
    Model.AddDescription(builder,desc)
    Model.AddOperatorCodes(builder,opCodesTFLite)
    Model.AddSubgraphs(builder,subGraphsTFLite)

    builder.Finish(Model.End(builder))

builder = fb.Builder(1024)

gen_Model(builder)

# gen_conv2d(builder)



buffer = builder.Output()

with open("test/out.tflite","wb") as f:
    f.write(buffer)