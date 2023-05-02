# ONNX to TFLite convertor

Open project to **directly** convert models from the *.onnx* to *.tflite* format. Currently a reduced subset of *ONNX* operators is supported.
Verified models:

* Alexnet (https://github.com/onnx/models/blob/main/vision/classification/alexnet/model/bvlcalexnet-12.onnx)
* TinyYOLO (https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-8.onnx)
* DUC (https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/duc/model/ResNet101-DUC-12.onnx)
* Speech Classifier (https://github.com/Pop-korn/ONNX2TFLite/blob/main/data/onnx/speech_command_classifier_trained.onnx)

The models can be downloaded using **make get-default-models** and then converted to *TFLite* using **make convert-default-models** from the root directory.

---

## Instalation

The program was developped using **python 3.10.8**. Some earlier versions will not work because of the use of type hints.

The program was developped and verified using **Ubuntu 22.04**. It should work on most Linux distributions.

The **pip** tool is required for instalation of Python packages.

To install just the Python packages required for model conversion, use **make install-essential** in the root directory. To install all used python packages (including the ones needed for testing), run **make install**. There is a requirement confilct between *tensorflow* and *onnx* over *protobuf* version. This program doesn't require that part of *tensorflow* functionality, so if there is a problem during instalation, run **pip install onnx==1.13.0**. Conversion testing should now work.

Project comes with *ONNX* and *TFLite* schemas pre-installed and libraries pre-compiled. If you want, you can optionally delete the *data/schemas* and *lib* directories and run **make regenerate-lib**  in the root directory to get the schemas and complie them yourself. THIS MAT CAUSE **make test-model-conversion** TO NOT WORK, because of limitations of the *onnx* library.

---

## Usage

**python onnx2tflite.py <input_model.onnx> [ -o/--output <output_model.tflite> --verbose]**

The program is used via a command line interface of the *onnx2tflite.py* module. It takes 1 required argument, which is the input *.onnx* model to be converted. The output *.tflite* file can be optionally specified using the **-o/--output** option. By default, it will be generated in the current working directory and have the same name as the input file. 

The program will only print error and warning messages, if the input model cannot be converted exactly. If you wish to see more detailed logging of the internal conversion process, use the **--verbose** option.

Example use:
* python onnx2tflite.py data/onnx/tinyyolov2-8.onnx --output test/tinyyolov2-8.tflite
* python onnx2tflite.py data/onnx/bvlcalexnet-12.onnx -o test/bvlcalexnet-12.tflite --verbose

---

## Testing

The module **conversion_test.py** implements functions for testing the accuracy of converted models. Tests for the default models are at the end of the file. Uncomment the lines which test the model you are interested in and run **make test-model-conversion** from the root directory.

Two testing functions are implemented. *runAndTestOperators()* allows testing of conversion of a selected portion of an *ONNX* model. It prints a short summary of the output of the original model, the output of the converted model and their difference. A more thorough testing function is *testConversion()*. It converts an entire *ONNX* model to *TFLite* and runs them on equal random data for multiple iterations. It prints a set of statistics describing the output errors.

The module *generator_test.py* implements a simple *.tflite* model in code. It generates and runs the model and prints its output. Also prints the output of the original model. Use **make test-tflite-file-generation** to run this test.

The *tensorflow* library sometimes prints error messages when loaded. This happens when GPU acceleration is not supported. These errors can be ignored.

---

## Structure
The entry point of the program is the *onnx2tflite.py* file in the root directory. 
The *conversion_test.py* modulde is used for testing of the accuracy of converted models.
The module *generator_test.py* implements a simple *.tflite* model in code and is used to test the **/src/generator** functionality.

The **/data** directory contains pre-trained models, tensor values, input images for inference testing and schemas. Some models are not included on GitHub because of size restrictions.


The code in the **/src** diroctory is split into multiple subdirectories.

* **/src/generator** contains classes used for internal representation of a TFLite model and for subsequent *.tflie* file generation.

* **/src/parser** conatins classes used for internal representation of an ONNX model and for loading it's data from a *.onnx* file.

* **/src/convertor** contains files for converting from the *parser* object model to the *generator* model.

The **/lib** directory contains files generated by compiling TFLite an ONNX schemas. They provide a low level interface for parsing a *.onnx* file and for generating a *.tflite* file.

* **/lib/tflite** is a library generated from the */data/schemas/tflite/schema.bfs* file, using the *flatc -p ../data/schema.fbs* command.

* **/lib/onnx** is a library generated from the files in the */data/schemas/onnx/* directory.

The **/test** directory is used for analyzing the generated *.tflite* models.

___

## License

This software is covered by the MIT license.
