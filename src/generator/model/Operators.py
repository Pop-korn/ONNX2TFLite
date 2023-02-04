import flatbuffers as fb

import tflite.SubGraph as sg
import tflite.Operator as op
import tflite.CustomOptionsFormat as cof

import generator.meta.meta as meta

class Inputs(meta.IntVector):
    def __init__(self, inputs: list[int]):
        super().__init__(inputs,op.StartInputsVector)

class Outputs(meta.IntVector):
    def __init__(self, outputs: list[int]):
        super().__init__(outputs,op.StartOutputsVector)

class MutatingVariableInputs(meta.BoolVector):
        def __init__(self, mutatingVariableInputs: list[bool]) -> None:
             super().__init__(mutatingVariableInputs,op.StartMutatingVariableInputsVector)


class Operator(meta.TFLiteObject):
    opcodeIndex: int
    customOptionsFormat: cof.CustomOptionsFormat # Only default value is possible
    mutatingVariableInputs: MutatingVariableInputs
    inputs: Inputs
    outputs: Outputs
    builtinOptions: meta.BuiltinOptions
    # TODO customOptions
    # TODO intermediates

    def __init__(self, inputs: Inputs, outputs: Outputs, builtinOptions: meta.BuiltinOptions
    , mutatingVariableInputs: MutatingVariableInputs, opcodeIndex: int = 0
    , customOptionsFormat: cof.CustomOptionsFormat = cof.CustomOptionsFormat.FLEXBUFFERS) -> None:
        self.opcodeIndex = opcodeIndex
        self.customOptionsFormat = customOptionsFormat
        self.mutatingVariableInputs = mutatingVariableInputs
        self.inputs = inputs
        self.outputs = outputs
        self.builtinOptions = builtinOptions

    def genTFLite(self, builder: fb.Builder):
        tflMutatingVariableInputs = self.mutatingVariableInputs.genTFLite(builder)
        tflInputs = self.inputs.genTFLite(builder)
        tflOutputs = self.outputs.genTFLite(builder)
        tflBuiltinOptions = self.builtinOptions.genTFLite(builder)

        op.Start(builder)

        op.AddOpcodeIndex(builder, self.opcodeIndex)
        op.AddMutatingVariableInputs(builder, tflMutatingVariableInputs)
        op.AddInputs(builder, tflInputs)
        op.AddOutputs(builder, tflOutputs)
        op.AddBuiltinOptions(builder, tflBuiltinOptions)
        op.AddBuiltinOptionsType(builder, self.builtinOptions.builtinOptionsType)

        return op.End(builder)


class Operators(meta.TFLiteVector):
    def __init__(self, operators: list[Operator]) -> None:
        super().__init__(operators,sg.StartOperatorsVector) # TODO UOffset??
        