import parser.model.Model as m

model = m.Model("data/onnx/bvlcalexnet-12.onnx")

print(model.opsetImport[0].version)

print(model.irVersion)
