.PHONY: all get_schema gen_json parse_json test

all:
	python3 ./src/main.py

get_schemas:
	wget -P ./data/schemas/tflite/ https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/schema/schema.fbs
	wget -P ./data/schemas/onnx/onnx/ https://raw.githubusercontent.com/onnx/onnx/main/onnx/onnx-ml.proto
	wget -P ./data/schemas/onnx/ https://raw.githubusercontent.com/onnx/onnx/main/onnx/onnx-data.proto
	wget -P ./data/schemas/onnx/ https://raw.githubusercontent.com/onnx/onnx/main/onnx/onnx-operators-ml.proto

test: all
	flatc -t --strict-json --defaults-json -o test data/schema.fbs -- test/out.tflite --raw-binary 

LB := (
RB := )
clear-pycache:
	find . | grep -E "$(LB)/__pycache__$$|\.pyc$$|\.pyo$$$(RB)" | xargs rm -rf
