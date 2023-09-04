from tensorflow.keras.layers import StringLookup
from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import os
import time
from google.cloud import storage
from tensorflow import keras
import tensorflow
import json
import cv2

from pgn_chess_games.model.registry import load_chess_interpreter, load_dictionary
from pgn_chess_games.model.data import (
    distortion_free_resize,
)

import re


img_for_box_extraction_path = "/root/code/laligb/pgn-chess-games/data/data/017_0.png"


def get_predictions(input_batch):
    interpreter = load_chess_interpreter()
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Ensure the input_batch has the correct shape
    if input_batch.shape[1:] != tuple(input_details[0]["shape"][1:]):
        raise ValueError(
            f"Input batch has shape {input_batch.shape[1:]} but model expects {input_details[0]['shape'][1:]}"
        )

    # Set the input tensor, invoke the interpreter, and get the output tensor for each image in the batch
    predictions = []
    for i in range(input_batch.shape[0]):
        input_data = np.expand_dims(input_batch[i], axis=0)
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]["index"])
        predictions.append(output_data)
    predictions = np.vstack(predictions)

    return predictions


model_properties_dict = load_dictionary("model_properties")
max_len = model_properties_dict["max_len"]
characters = model_properties_dict["characters"]

# Mapping characters to integers.
char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)

# Mapping integers back to original characters.
num_to_char = StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

LOCAL_DATA_PATH = os.environ["LOCAL_DATA_PATH"]
PRED_PATH = os.path.join(LOCAL_DATA_PATH, "prediction/")


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(
        *sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse)
    )

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def box_extraction(img_for_box_extraction_path, cropped_dir_path):
    print("Reading image..")
    img = cv2.imread(img_for_box_extraction_path, 0)  # Read the image
    (thresh, img_bin) = cv2.threshold(
        img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )  # Thresholding the image
    img_bin = 255 - img_bin  # Invert the image

    print("Storing binary image to Images/Image_bin.jpg..")
    cv2.imwrite("Images/Image_bin.jpg", img_bin)

    print("Applying Morphological Operations..")
    # Defining a kernel length
    kernel_length = np.array(img).shape[1] // 40

    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Morphological operation to detect verticle lines from an image
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
    cv2.imwrite("Images/verticle_lines.jpg", verticle_lines_img)

    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
    cv2.imwrite("Images/horizontal_lines.jpg", horizontal_lines_img)

    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha
    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(
        verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0
    )
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(
        img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )

    # For Debugging
    # Enable this line to see verticle and horizontal lines in the image which is used to find boxes
    print("Binary image which only contains boxes: Images/img_final_bin.jpg")
    cv2.imwrite("Images/img_final_bin.jpg", img_final_bin)
    # Find contours for image, which will detect all the boxes
    contours, hierarchy = cv2.findContours(
        img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    # Sort all the contours by top to bottom.
    (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")

    print("Output stored in Output directiory!")

    idx = 0
    for c in contours:
        # Returns the location and width,height for every contour
        x, y, w, h = cv2.boundingRect(c)

        # If the box height is greater then 20, widht is >80, then only save it as a box in "cropped/" folder.
        if (w > 80 and h > 20) and w > 3 * h:
            idx += 1
            new_img = img[y : y + h, x : x + w]
            cv2.imwrite(cropped_dir_path + str(idx) + ".png", new_img)


box_extraction(img_for_box_extraction_path, PRED_PATH)

batch_size = 180
padding_token = 99
image_width = 128
image_height = 32

prediction = True

AUTOTUNE = tf.data.AUTOTUNE


def preprocess_image(
    image_path, img_size=(image_width, image_height), prediction=False
):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, 1)
    image = distortion_free_resize(image, img_size, prediction)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def vectorize_label(label):
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    length = tf.shape(label)[0]
    pad_amount = max_len - length
    label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=padding_token)
    return label


def process_images_labels(image_path, label):
    image = preprocess_image(image_path)
    label = vectorize_label(label)
    return {"image": image, "label": label}


def prepare_dataset(image_paths, labels, prediction=False):
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            image_paths,
            labels,
        )
    ).map(process_images_labels, num_parallel_calls=AUTOTUNE)
    return dataset.batch(batch_size).prefetch(AUTOTUNE)


def create_predict_ds(path):
    predict_image_folder = os.listdir(PRED_PATH)
    predict_image_paths = [os.path.join(PRED_PATH, x) for x in predict_image_folder]
    predict_image_paths.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))

    pred_ds = prepare_dataset(
        predict_image_paths, ["blank" for each in predict_image_paths]
    )
    return pred_ds


def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = tensorflow.keras.backend.ctc_decode(
        pred, input_length=input_len, greedy=True
    )[0][0][:, :21]

    char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)

    num_to_char = StringLookup(
        vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    )

    output_text = []
    for res in results:
        res = tensorflow.gather(
            res, tensorflow.where(tensorflow.math.not_equal(res, -1))
        )
        res = tensorflow.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


def predict_batch(path):
    pred_ds = create_predict_ds(path)
    for batch in pred_ds:
        batch_images = batch["image"]
        predictions = get_predictions(batch_images)
        texts = decode_batch_predictions(predictions)
    print(texts)
    return texts
