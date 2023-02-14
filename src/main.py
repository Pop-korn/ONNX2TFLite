import parser.model.Model as m

model = m.Model("data/onnx/bvlcalexnet-12.onnx")

print(model.modelVersion)