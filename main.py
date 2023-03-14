import src.converter.convert as convert

""" Temporary. Handle which models to convert. """
# convert.convertModel("data/onnx/bvlcalexnet-12.onnx", "test/alexnet.tflite")
convert.convertModel("data/onnx/tinyyolov2-8.onnx", "test/tinyyolo.tflite")

