import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow
from tensorflow.keras.layers import StringLookup
from tensorflow import keras

## Setting a set random seed to ensure consistency of shuffle
np.random.seed(42)
tensorflow.random.set_seed(42)

## Local data path where dataset is stored
LOCAL_DATA_PATH = os.path.join(os.environ["LOCAL_DATA_PATH"], "words")


def define_data(path):
    ## Defining the shuffled dataset
    words_list = []
    words = open(f"{LOCAL_DATA_PATH}/words.txt", "r").readlines()
    for line in words:
        if line[0] == "#":
            continue
        if line.split(" ")[1] != "err":  # We don't need to deal with errored entries.
            words_list.append(line)
    np.random.shuffle(words_list)
    return words_list


def database_split(words_list):
    """Obtaining the dataset from path provided,
    then splitting into train, test and validation in 90:5:5 ratio
    Return 3 buckets of train, val and test data"""

    split_idx = int(0.9 * len(words_list))
    data_train = words_list[:split_idx]
    data_test = words_list[split_idx:]

    val_split_idx = int(0.5 * len(data_test))
    data_val = data_test[:val_split_idx]
    data_test = data_test[val_split_idx:]

    assert len(words_list) == len(data_train) + len(data_test) + len(data_val)

    return data_train, data_val, data_test


def get_image_paths_and_labels(samples):
    LOCAL_DATA_PATH = os.path.join(os.environ["LOCAL_DATA_PATH"], "words")
    paths = []
    corrected_samples = []
    for i, file_line in enumerate(samples):
        line_split = file_line.strip()
        line_split = line_split.split(" ")

        # Each line split will have this format for the corresponding image:
        # part1/part1-part2/part1-part2-part3.png
        image_name = line_split[0]
        partI = image_name.split("-")[0]
        partII = image_name.split("-")[1]
        img_path = os.path.join(
            LOCAL_DATA_PATH, partI, partI + "-" + partII, image_name + ".png"
        )
        if os.path.getsize(img_path):
            paths.append(img_path)
            corrected_samples.append(file_line.split("\n")[0])

    return paths, corrected_samples


def get_data(words_list):
    """Return three datasets: train, validation and test in ratio of
    90:5:5 to be used for the model training"""

    ## Create train, val and test data based on shuffled database
    data_train, data_val, data_test = database_split(words_list)
    return data_train, data_val, data_test


def get_constants(validation_ds):
    ## Compute vocabulary size of the dataset
    characters = set()
    max_len = 0
    for label in validation_ds:
        label = label.split(" ")[-1].strip()
    for char in label:
        characters.add(char)
    max_len = max(max_len, len(label))
    characters = sorted(list(characters))

    return characters, max_len


def cleaning_labels(labels):
    ## Clean the labels
    clean_labels = []
    for label in labels:
        label = label.split(" ")[-1].strip()
        clean_labels.append(label)

    return clean_labels


def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tensorflow.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Check tha amount of padding needed to be done.
    pad_height = h - tensorflow.shape(image)[0]
    pad_width = w - tensorflow.shape(image)[1]

    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tensorflow.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )

    image = tensorflow.transpose(image, perm=[1, 0, 2])
    image = tensorflow.image.flip_left_right(image)
    return image
