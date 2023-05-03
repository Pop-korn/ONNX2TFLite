TFL?=test.tflite
JSON?=test.json
ONNX?=test.onnx

# Convert a .tflite file to json. Some models are too big!
flatbuffer-to-json:
	flatc -t --strict-json --defaults-json -o . ../data/schemas/tflite/schema.fbs -- $(TFL) --raw-binary

# Convert a json file to .tflite
json-to-flatbuffer:
	flatc -b --strict-json --defaults-json -o . ../data/schemas/tflite/schema.fbs $(JSON)

# Convert a .onnx file to json. Somve models are too big!
protobuf-to-json:
	python protobufToJson.py $(ONNX) > $(JSON)
	
	
# Test if generated .tflite file is smaller than one created by flatc through json
test-efficiency: 
	flatc -t --strict-json --defaults-json -o . ../data/schemas/tflite/schema.fbs -- $(TFL) --raw-binary
	mv $(TFL) old.tflite
	flatc -b --strict-json --defaults-json -o . ../data/schemas/tflite/schema.fbs $(JSON)
	diff $(TFL) old.tflite
	
# Remove all .onnx, .tflite and .json files	
clear:
	rm *.tflite *.onnx *.json 2> /dev/null
	
