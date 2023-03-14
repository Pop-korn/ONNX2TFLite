import src.converter.convert as convert

""" Temporary. Handle which models to convert. """
# convert.convertModel("data/onnx/bvlcalexnet-12.onnx", "test/alexnet.tflite")
convert.convertModel("data/onnx/ssd_mobilenet_v1_12.onnx", "test/mobilenet.tflite")

