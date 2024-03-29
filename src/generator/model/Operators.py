"""
    Operators

Module contains classes that represent TFLite 'Operator' objects.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

from typing import List
import flatbuffers as fb

import lib.tflite.SubGraph as sg
import lib.tflite.Operator as op
import lib.tflite.CustomOptionsFormat as cof

import src.generator.meta.meta as meta
import src.generator.model.Tensors as Tensors

class Inputs(meta.IntVector):
    def __init__(self, inputs: List[int] = None):
        super().__init__(inputs,op.StartInputsVector)

class Outputs(meta.IntVector):
    def __init__(self, outputs: List[int] = None):
        super().__init__(outputs,op.StartOutputsVector)

class MutatingVariableInputs(meta.BoolVector):
    def __init__(self, mutatingVariableInputs: List[bool] = None) -> None:
        super().__init__(mutatingVariableInputs, op.StartMutatingVariableInputsVector)


class Operator(meta.TFLiteObject):
    opcodeIndex: int
    customOptionsFormat: cof.CustomOptionsFormat # Only default value is possible
    mutatingVariableInputs: MutatingVariableInputs
    inputs: Inputs
    outputs: Outputs
    builtinOptions: meta.BuiltinOptions
    # TODO customOptions
    # TODO intermediates


    """ IMPORTANT! The following attributes are used only by 'ModelBuilder' 
        in order to make model creation more eficient. """

    """ Lists of references to 'Tensor' objects. Simpler to use when converting
        than 'inputs' and 'outputs'. """
    tmpInputs: List[Tensors.Tensor]
    tmpOutputs: List[Tensors.Tensor]

    def __init__(self, inputs: Inputs=None, outputs: Outputs=None,
                builtinOptions: meta.BuiltinOptions=None,
                opcodeIndex: int = 0, 
                mutatingVariableInputs: MutatingVariableInputs=MutatingVariableInputs(),
                customOptionsFormat: cof.CustomOptionsFormat = cof.CustomOptionsFormat.FLEXBUFFERS) -> None:
        self.opcodeIndex = opcodeIndex
        self.customOptionsFormat = customOptionsFormat
        self.mutatingVariableInputs = mutatingVariableInputs
        self.builtinOptions = builtinOptions
        if inputs is None:
            inputs = Inputs()
        self.inputs = inputs
        if outputs is None:
            outputs = Outputs()
        self.outputs = outputs

        self.tmpInputs = []
        self.tmpOutputs = []

    def genTFLite(self, builder: fb.Builder):
        if self.inputs is not None:
            tflInputs = self.inputs.genTFLite(builder)

        if self.outputs is not None:
            tflOutputs = self.outputs.genTFLite(builder)

        if self.builtinOptions is not None:
            tflBuiltinOptions = self.builtinOptions.genTFLite(builder)

        if self.mutatingVariableInputs is not None:
            tflMutatingVariableInputs = self.mutatingVariableInputs.genTFLite(builder)


        op.Start(builder)

        op.AddOpcodeIndex(builder, self.opcodeIndex)

        if self.inputs is not None:
            op.AddInputs(builder, tflInputs)

        if self.outputs is not None:
            op.AddOutputs(builder, tflOutputs)

        if self.builtinOptions is not None:
            op.AddBuiltinOptions(builder, tflBuiltinOptions)
            op.AddBuiltinOptionsType(builder, self.builtinOptions.builtinOptionsType)

        if self.mutatingVariableInputs is not None:
            op.AddMutatingVariableInputs(builder, tflMutatingVariableInputs)
            
        return op.End(builder)


class Operators(meta.TFLiteVector):
    def __init__(self, operators: List[Operator] = None) -> None:
        super().__init__(operators,sg.StartOperatorsVector)
        