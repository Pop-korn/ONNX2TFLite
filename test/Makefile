to-json:
	flatc -t --strict-json --defaults-json -o . ../data/schemas/tflite/schema.fbs -- cifar10_model_GENERATED.tflite --raw-binary

from-json:
	flatc -b --strict-json --defaults-json -o . ../data/schemas/tflite/schema.fbs cifar10_model_GENERATED.json
	
