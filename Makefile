.PHONY: all get_schema gen_json parse_json test

all:
	python3 main.py

get_schema:
	wget https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/schema/schema.fbs
gen_json:
	flatc -t --strict-json --defaults-json schema.fbs -- cifar10_model.tflite
parse_json:
	flatc -b --strict-json --defaults-json -o . schema.fbs cifar10_model.json

test:
	flatc -t --strict-json --defaults-json schema.fbs -- test/out.tflite --raw-binary