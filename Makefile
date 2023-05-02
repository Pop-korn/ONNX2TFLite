# Select preferred python interpreter
PYTHON := python3

.PHONY: install test-model-conversion get-default-models convert-default-models 
.PHONY: get-schemas compile-tflite-schema compile-onnx-schema regenrate-lib 
.PHONY: test-tflite-file-generation clear-pycache total-line-count

# Install all python packages rquired for model conversion
install-essential:
	pip install -r requirements_essential.txt

# Install all python packages, required for model conversion as well as testing
install:
	pip install -r requirements.txt

# Test the accuracy of converted models.
# Uncomment lines at the end of 'conversion_test.py' to specify tests and models
test-model-conversion: 
	$(PYTHON) conversion_test.py

# Download the verified .onnx models into the 'data/onnx/' directory
get-default-models:
	wget -P ./data/onnx/ https://github.com/onnx/models/raw/main/vision/classification/alexnet/model/bvlcalexnet-12.onnx
	wget -P ./data/onnx/ https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-8.onnx
	wget -P ./data/onnx/ https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/duc/model/ResNet101-DUC-12.onnx

# Convert all verified .onnx models. Outputs are in the 'test/' directory
convert-default-models:
	$(PYTHON) onnx2tflite.py data/onnx/bvlcalexnet-12.onnx -o test/bvlcalexnet-12.tflite
	$(PYTHON) onnx2tflite.py data/onnx/ResNet101-DUC-12.onnx -o test/ResNet101-DUC-12.tflite
	$(PYTHON) onnx2tflite.py data/onnx/tinyyolov2-8.onnx -o test/tinyyolov2-8.tflite
	$(PYTHON) onnx2tflite.py data/onnx/speech_command_classifier_trained.onnx -o test/speech_command_classifier_trained.tflite



# Download the ONNX and TFLite model schemas
get-schemas:
	wget -P ./data/schemas/tflite/ https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/schema/schema.fbs
	wget -P ./data/schemas/onnx/onnx/ https://raw.githubusercontent.com/onnx/onnx/main/onnx/onnx-ml.proto
	wget -P ./data/schemas/onnx/ https://raw.githubusercontent.com/onnx/onnx/main/onnx/onnx-data.proto
	wget -P ./data/schemas/onnx/ https://raw.githubusercontent.com/onnx/onnx/main/onnx/onnx-operators-ml.proto

# Compile the schemas to generate flatbuffer and protobuf libraries.
compile-tflite-schema:
	flatc -p -o ./lib/ data/schemas/tflite/schema.fbs
compile-onnx-schema:
	mkdir -p lib
	mkdir -p lib/onnx
	cd data/schemas/onnx ; protoc --python_out=../../../lib/onnx onnx/onnx-ml.proto onnx-operators-ml.proto onnx-data.proto

# Download the schemas and compile them to create the flatbuffer and protobuf libraries
regenrate-lib: get-schemas compile-tflite-schema compile-onnx-schema

# Generate a simple TFLite model from code and run it. Print its and the original models output
test-tflite-file-generation:
	$(PYTHON) ./generator_test.py

LB := (
RB := )
# Delete pycache files
clear-pycache:
	find . | grep -E "$(LB)/__pycache__$$|\.pyc$$|\.pyo$$$(RB)" | xargs rm -rf

# Count the total number of lines in the program
total-line-count: clear-pycache
	find src/ | xargs wc -l 
