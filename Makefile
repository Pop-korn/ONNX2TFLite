.PHONY: all get_schema gen_json parse_json test

all:
	python3 ./src/main.py

get_schema:
	wget -P ./data/ https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/schema/schema.fbs

test: all
	flatc -t --strict-json --defaults-json -o test data/schema.fbs -- test/out.tflite --raw-binary 

LB := (
RB := )
clear-pycache:
	find . | grep -E "$(LB)/__pycache__$$|\.pyc$$|\.pyo$$$(RB)" | xargs rm -rf