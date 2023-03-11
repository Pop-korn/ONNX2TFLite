TFL?=alexnet_reduced.tflite

to-json:
	flatc -t --strict-json --defaults-json -o . ../data/schemas/tflite/schema.fbs -- $(TFL) --raw-binary

from-json:
	flatc -b --strict-json --defaults-json -o . ../data/schemas/tflite/schema.fbs cifar10_model_GENERATED.json
	
