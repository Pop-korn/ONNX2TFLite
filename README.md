# ONNX to TFLite convertor
## **Work In Progress**
Open project to convert models from the *.onnx* to *.tflite* format.

Currently *.tflite* model generation from code, *.onnx* model loading and conversion of tensors are supported. Working on converting operators.

Project should be in a usable state by June 2023.

---
## Structure
The entry point of the program is the *main.py* file in the root directory. To run the program use Makefile in the root directory. A *.tflite* model will be generated in the **/test** directory.

The **/data** directory contains pre-trained models, tensor values input images for inference testing and schemas. Some models are not included on GitHub because of size restrictions.


The code in the **/src** diroctory is split into multiple subdirectories.

* **/src/generator** contains classes used for internal representation of a TFLite model and for subsequent *.tflie* file generation.

* **/src/parser** conatins classes used for internal representation of an ONNX model and for loading it's data from a *.onnx* file.

* **/src/convertor** contains files for convertin from the *parser* object model to the *generator* model.

The **/lib** directory contains files generated by compiling TFLite an ONNX schemas. They provide a low level interface for parsing a *.onnx* file and for generating a *.tflite* file.

* **/lib/tflite** is a library generated from the */data/schemas/tflite/schema.bfs* file, using the *flatc -p ../data/schema.fbs* command.

* **/lib/onnx** is a library generated from the files in the */data/schemas/onnx/* directory.

The **/test** directory is used for analyzing the generated *.tflite* models.



