.PHONY: all get_schema gen_json parse_json test

all:
	python3 ./main.py

install:
	pip install -r requirements.txt

test:
	python3 cvt_test.py

to-json: all
	flatc -t --strict-json --defaults-json -o test data/schemas/tflite/schema.fbs -- test/alexnet.tflite --raw-binary 

generator-test:
	export TF_CPP_MIN_LOG_LEVEL="2"
	python3 ./generator_test.py

get-schemas:
	wget -P ./data/schemas/tflite/ https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/schema/schema.fbs
	wget -P ./data/schemas/onnx/onnx/ https://raw.githubusercontent.com/onnx/onnx/main/onnx/onnx-ml.proto
	wget -P ./data/schemas/onnx/ https://raw.githubusercontent.com/onnx/onnx/main/onnx/onnx-data.proto
	wget -P ./data/schemas/onnx/ https://raw.githubusercontent.com/onnx/onnx/main/onnx/onnx-operators-ml.proto
compile-tflite-schema:
	flatc -p -o ./lib/ data/schemas/tflite/schema.fbs
compile-onnx-schema:
	mkdir -p lib
	mkdir -p lib/onnx
	cd data/schemas/onnx ; protoc --python_out=../../../lib/onnx onnx/onnx-ml.proto onnx-operators-ml.proto onnx-data.proto
regenrate-lib: get-schemas compile-tflite-schema compile-onnx-schema




LB := (
RB := )
clear-pycache:
	find . | grep -E "$(LB)/__pycache__$$|\.pyc$$|\.pyo$$$(RB)" | xargs rm -rf

total-line-count:
	find src/ | xargs wc -l
