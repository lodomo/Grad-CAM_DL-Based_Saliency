###############################################################################
#
# Author: Lorenzo D. Moon
# Professor: Anthony Rhodes
# Course: CS-410 : Computer Vision & Deep Learning
# Assignment: Programming Assignment 4
# Description: Exercise #2: Grad-CAM DL-Based Saliency
#
###############################################################################

import sys

import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K
from tensorflow.compat.v1.keras.applications import VGG16
from tensorflow.compat.v1.keras.applications.vgg16 import (decode_predictions,
                                                           preprocess_input)
from tensorflow.compat.v1.keras.preprocessing import image


def main(argv):
    tf.compat.v1.disable_eager_execution()  # Allows K.gradients to work
    img_path, img_name, prediction_level = process_command_line(argv)
    model = VGG16(weights="imagenet")
    og_img = cv2.imread(img_path)  # Hold the original for later
    img = process_image(img_path)  # Process for VGG16

    # Get the top 3 predictions from the model.
    predictions = model.predict(img)
    top_3 = decode_predictions(predictions, top=3)[0]  # VGG16 Library Function

    # Grab the top prediction (or the one specified by the user)
    chosen_pred = top_3[prediction_level]
    class_index = np.argmax(predictions[0] == chosen_pred[2])

    # Get the last convolutional layer in the model
    last_layer = model.get_layer("block5_conv3").output

    # Compute the gradient of the class output value with respect to the
    # feature map using the built-in Keras function
    gradient = K.gradients(model.output[:, class_index], last_layer)[0]

    # Compute the mean of the gradient over the feature map
    # This is performing the a_k^c from the Grad-CAM paper
    # This is still in the Keras backend, and needs to be extracted
    akc = K.mean(gradient, axis=(0, 1, 2))

    # Defines a function to pull the values of the last layer and akc values
    extract_values = K.function([model.input], [akc, last_layer[0]])

    # Get the actual values of the pooled gradients and the last layer
    akc_val, Ak = extract_values([img])

    # This is performing the a_k^c * A^k portion
    # Multiply each channel in the feature map array by the gradient value
    for i in range(akc.shape[-1]):
        Ak[:, :, i] *= akc_val[i]

    # Sum the conv layer output value to get the saliency map
    saliency_map = np.sum(Ak, axis=-1)

    # ReLU activation
    saliency_map = np.maximum(saliency_map, 0)
    # Now the L^cGrad-Cam is complete.

    # Normalize the calculated saliency map
    saliency_map /= np.max(saliency_map)

    # Keep an unscaled copy of the saliency map for saving
    raw_map = np.copy(saliency_map)
    raw_map = np.uint8(255 * raw_map)
    raw_map = cv2.resize(raw_map, (500, 500), interpolation=cv2.INTER_NEAREST)
    raw_map = cv2.applyColorMap(raw_map, cv2.COLORMAP_JET)

    # Resize the saliency map to the size of the original image,
    # and apply a color map COLORMAP_JET seemed to match the assignment example
    # Overlay the saliency map on the original image
    saliency_map = cv2.resize(saliency_map, (og_img.shape[1], og_img.shape[0]))
    saliency_map = np.uint8(255 * saliency_map)
    saliency_map = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
    super_img = apply_saliency_map(og_img, saliency_map, 0.6)

    # Save images
    save_pre = f"./output/{img_name}_{prediction_level + 1}_{chosen_pred[1]}"
    raw_saliency_map = save_pre + "_raw_saliency.jpg"
    saliency_map_path = save_pre + "_saliency.jpg"
    super_path = save_pre + "_superimposed.jpg"
    perror(f"Raw saliency saved to {raw_saliency_map}")
    cv2.imwrite(raw_saliency_map, raw_map)
    perror(f"Saliency map saved to {saliency_map_path}")
    cv2.imwrite(saliency_map_path, saliency_map)
    perror(f"Superimposed image saved to {super_path}")
    cv2.imwrite(super_path, super_img)

    prediction = f"Image: {img_name} | Prediction: {chosen_pred[1]} "
    prediction += f"| Confidence: {chosen_pred[2]}"
    print(prediction)
    exit(0)


def process_command_line(argv):
    # One arg, filename of image
    if len(argv) < 1:
        print("Usage: python3 main.py <image_filename> <prediction level>")
        sys.exit(1)

    prediction_level = 1
    # Prediction level
    if len(argv) == 2:
        prediction_level = int(argv[1])

    if prediction_level < 1 or prediction_level > 3:
        print("Prediction level must be between 1 and 3")
        sys.exit(1)

    # Since the predictions are 0-indexed, decrement the level
    prediction_level -= 1

    # Get the image path and name
    img_path = argv[0]
    img_name = img_path.split("/")[-1]
    img_name = img_name.split(".")[0]

    return img_path, img_name, prediction_level


def apply_saliency_map(og_img, saliency_map, intensity):
    # Apply the saliency map to the original image
    # I found that the intensity of 0.6 was a good balance
    # between the original image and the saliency map
    superimposed_img = cv2.addWeighted(
        og_img, 1 - intensity, saliency_map, intensity, 0
    )
    return superimposed_img


def process_image(img_path):
    # Preprocess image for VGG16
    # Load image, convert to an array, and preprocess 
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)

    # Expand the dimensions of the image to fix this error:
    # ValueError: Error when checking input: expected input_1 to have 4
    # dimensions, but got array with shape (224, 224, 3)
    img = np.expand_dims(img, axis=0)

    img = preprocess_input(img)  # VGG16 Library Function
    return img


def perror(msg):
    # Prevent messages from being saved when sending stdout to file.
    print(msg, file=sys.stderr)
    return


if __name__ == "__main__":
    main(sys.argv[1:])
