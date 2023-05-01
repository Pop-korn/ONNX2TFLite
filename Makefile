.PHONY: all install test-model-conversion get-default-models convert-default-models 
.PHONY: get-schemas compile-tflite-schema compile-onnx-schema regenrate-lib test-tflite-file-generation clear-pycache total-line-count

all: # Basic
	python ./onnx2tflite.py data/onnx/bvlcalexnet-12.onnx --output test/alexnet.tflite

install:
	pip install -r requirements.txt

test-model-conversion:
	python conversion_test.py

get-default-models:
	wget -P ./data/onnx/ https://github.com/onnx/models/raw/main/vision/classification/alexnet/model/bvlcalexnet-12.onnx
	wget -P ./data/onnx/ https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-8.onnx
	wget -P ./data/onnx/ https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/duc/model/ResNet101-DUC-12.onnx

convert-default-models:
	python onnx2tflite.py data/onnx/bvlcalexnet-12.onnx -o test/bvlcalexnet-12.tflite
	python onnx2tflite.py data/onnx/ResNet101-DUC-12.onnx -o test/ResNet101-DUC-12.tflite
	python onnx2tflite.py data/onnx/tinyyolov2-8.onnx -o test/tinyyolov2-8.tflite
	python onnx2tflite.py data/onnx/speech_command_classifier_trained.onnx -o test/speech_command_classifier_trained.tflite

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

test-tflite-file-generation:
	python ./generator_test.py

LB := (
RB := )
clear-pycache:
	find . | grep -E "$(LB)/__pycache__$$|\.pyc$$|\.pyo$$$(RB)" | xargs rm -rf

total-line-count:
	find src/ | xargs wc -l 
