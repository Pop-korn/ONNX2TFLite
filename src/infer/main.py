import tensorflow as tf
import numpy as np

from PIL import Image

my_inter = tf.lite.Interpreter(model_path = "../../test/out.tflite")
my_inter.allocate_tensors()

cifar_inter = tf.lite.Interpreter(model_path = "../../data/cifar10_model.tflite")
cifar_inter.allocate_tensors()


image = Image.open("../../data/cifar10/airplane1.png")
image = [np.asarray(image).tolist()]
image = np.asarray(image,np.uint8)

print(image.shape)
# print(image)

def classify(interpreter, image):
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)


    

classify(my_inter, image)
classify(cifar_inter, image)
