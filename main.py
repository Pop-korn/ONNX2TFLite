import src.parser.model.Model as m

import src.convertor.convert as convert

model = m.Model("data/onnx/bvlcalexnet-12.onnx")

convert.convertModel(model)
