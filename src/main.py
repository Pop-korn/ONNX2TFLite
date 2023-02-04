import flatbuffers as fb
import tflite.Model as Model
import tflite.BuiltinOperator as BuiltinOperator
import tflite.TensorType as TensorType

import generator.model.OperatorCodes as OperatorCodes
import generator.model.SubGraphs as SubGraphs
import generator.model.Tensors as Tensors
import generator.model.Quantization as Quant
import generator.model.Operators as Operators

def gen_Model(builder: fb.Builder):
    desc = builder.CreateString("Model Description")

    # Operator Codes
    operatorCodes = [OperatorCodes.OperatorCode(BuiltinOperator.BuiltinOperator.CONV_2D)]
    opCodesTFLite = OperatorCodes.genOperatorCodes(builder,operatorCodes)

    # SubGraphs
    quantization = Quant.Quantisation(Quant.Min([0.0]), Quant.Max([0.996094])
                    , Quant.Scale([0.003906]), Quant.ZeroPoint([0]), 0)
    tensors = Tensors.Tensors([Tensors.Tensor(quantization, Tensors.Shape([1,10]), 
    "CifarNet/Predictions/Reshape_1",17,TensorType.TensorType.UINT8)])

    inputs = SubGraphs.Inputs([0])
    outputs = SubGraphs.Outputs([2,3,5])

    operators = Operators.Operators([Operators.Operator(Operators.Inputs([16,3,1]),Operators.Outputs([2]),
        Operators.MutatingVariableInputs([]))])

    subGraphs = [SubGraphs.SubGraph(inputs, outputs, tensors,operators)]
    subGraphsTFLite = SubGraphs.genSubGraphs(builder,subGraphs)

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