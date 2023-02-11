# ONNX to TFLite convertor
## **Work In Progress**
Open project to convert models from the *.onnx* to *.tflite* format.

Currently only *.tflite* model generation from code is supported.

Project should be in a usable state by June 2023.

---
## Structure
To run the program use Makefile in the root directory. A *.tflite* model will be generated in the **/test** directory.

The **/data** directory contains pre-trained models, tensor values and input images for inference testing.

The code in the **/src** diroctory is split into multiple subdirectories.

* **/src/generator** contains classes used for internal representation of a TFLite model and for subsequent *.tflie* file generation.
* **/src/tflite** is a library generated from the */data/schema.bfs* file, using the *flatc -p ../data/schema.fbs* command.
* **/src/infer** contains files for testing of the generated models. Contains it's own Makefile.

The **/test** directory is used for analyzing the generated *.tflite* models.



