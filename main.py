import flatbuffers as fb

import tflite.Model as model
import tflite.Conv2DOptions as conv2d

import tflite.ActivationFunctionType as actFunType
import tflite.Padding as padding

def gen_conv2d(builder: fb.Builder):
    conv2d.Start(builder)

    conv2d.AddDilationHFactor(builder,1)
    conv2d.AddDilationWFactor(builder,1)

    conv2d.AddFusedActivationFunction(builder,actFunType.ActivationFunctionType.NONE)

    conv2d.AddPadding(builder,padding.Padding.SAME)

    conv2d.AddStrideH(builder,1)
    conv2d.AddStrideW(builder,1)

    conv = conv2d.End(builder)

    builder.Finish(conv)


def gen_model(builder: fb.Builder):
    model.Start(builder)

    model.AddVersion(builder,1)
    model.AddOperatorCodes(builder,)

    builder.Finish(model.End(builder))



builder = fb.Builder(1024)

gen_model(builder)

gen_conv2d(builder)

buffer = builder.Output()

with open("test/out.tflite","wb") as f:
    f.write(buffer)