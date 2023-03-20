TFL?=test.tflite
JSON?=test.json

to-json:
	flatc -t --strict-json --defaults-json -o . ../data/schemas/tflite/schema.fbs -- $(TFL) --raw-binary

from-json:
	flatc -b --strict-json --defaults-json -o . ../data/schemas/tflite/schema.fbs $(JSON)
	
	
test-efficiency: 
	flatc -t --strict-json --defaults-json -o . ../data/schemas/tflite/schema.fbs -- $(TFL) --raw-binary
	mv $(TFL) old.tflite
	flatc -b --strict-json --defaults-json -o . ../data/schemas/tflite/schema.fbs $(JSON)
	diff $(TFL) old.tflite
		
