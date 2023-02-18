.PHONY: all get_schema gen_json parse_json test

all:
	python3 ./main.py

test: all
	flatc -t --strict-json --defaults-json -o test data/schemas/tflite/schema.fbs -- test/alexnet.tflite --raw-binary 

generator_test:
	export TF_CPP_MIN_LOG_LEVEL="2"
	python3 ./generator_test.py

get_schemas:
	wget -P ./data/schemas/tflite/ https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/schema/schema.fbs
	wget -P ./data/schemas/onnx/onnx/ https://raw.githubusercontent.com/onnx/onnx/main/onnx/onnx-ml.proto
	wget -P ./data/schemas/onnx/ https://raw.githubusercontent.com/onnx/onnx/main/onnx/onnx-data.proto
	wget -P ./data/schemas/onnx/ https://raw.githubusercontent.com/onnx/onnx/main/onnx/onnx-operators-ml.proto

LB := (
RB := )
clear-pycache:
	find . | grep -E "$(LB)/__pycache__$$|\.pyc$$|\.pyo$$$(RB)" | xargs rm -rf
