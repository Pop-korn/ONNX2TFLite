TFL?=test.tflite
JSON?=test.json
ONNX?=test.onnx

flatbuffer-to-json:
	flatc -t --strict-json --defaults-json -o . ../data/schemas/tflite/schema.fbs -- $(TFL) --raw-binary

json-to-flatbuffer:
	flatc -b --strict-json --defaults-json -o . ../data/schemas/tflite/schema.fbs $(JSON)

protobuf-to-json:
	python protobufToJson.py $(ONNX) > $(JSON)
	
	
test-efficiency: 
	flatc -t --strict-json --defaults-json -o . ../data/schemas/tflite/schema.fbs -- $(TFL) --raw-binary
	mv $(TFL) old.tflite
	flatc -b --strict-json --defaults-json -o . ../data/schemas/tflite/schema.fbs $(JSON)
	diff $(TFL) old.tflite
		
