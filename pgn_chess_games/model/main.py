import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow
from tensorflow.keras.layers import StringLookup
from pgn_chess_games.model.data import *
from pgn_chess_games.model.model import *
from pgn_chess_games.model.callback import *
from tensorflow import keras

## Setting a set random seed to ensure consistency of shuffle
np.random.seed(42)
tensorflow.random.set_seed(42)

## Local data path where dataset is stored

CHARACTERS = None
MAX_LEN = 0


def train_model(train_ds, validation_ds, test_ds, epochs=10):
    size = (128, 32)
    model = initialize_model(size)
    prediction_model = keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(name="dense2").output
    )
    edit_distance_callback = EditDistanceCallback(prediction_model, validation_ds)

    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=epochs,
        callbacks=[edit_distance_callback],
    )
    return model, history


def decode_batch_predictions(pred, characters):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :21
    ]

    # Iterate over the results and get back the text.
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


def preprocess_image(image_path, img_size=(128, 32)):
    image = tensorflow.io.read_file(image_path)
    image = tensorflow.image.decode_png(image, 1)
    image = distortion_free_resize(image, img_size)
    image = tensorflow.cast(image, tensorflow.float32) / 255.0
    return image


def vectorize_label(label, padding_token=99):
    # Mapping characters to integers.
    char_to_num = StringLookup(vocabulary=list(CHARACTERS), mask_token=None)

    label = char_to_num(tensorflow.strings.unicode_split(label, input_encoding="UTF-8"))
    length = tensorflow.shape(label)[0]
    pad_amount = MAX_LEN - length
    label = tensorflow.pad(
        label, paddings=[[0, pad_amount]], constant_values=padding_token
    )
    return label


def process_images_labels(image_path, label):
    image = preprocess_image(image_path)
    label = vectorize_label(label)
    return {"image": image, "label": label}


def prepare_dataset(image_paths, labels, batch_size=64):
    AUTOTUNE = tensorflow.data.AUTOTUNE
    dataset = tensorflow.data.Dataset.from_tensor_slices((image_paths, labels)).map(
        process_images_labels, num_parallel_calls=AUTOTUNE
    )
    return dataset.batch(batch_size).cache().prefetch(AUTOTUNE)


def main():
    ## Define the data with shuffle function
    LOCAL_DATA_PATH = os.path.join(os.environ["LOCAL_DATA_PATH"], "words")

    words_list = define_data(LOCAL_DATA_PATH)
    print(f"✅ Shuffled data created withs size: {len(words_list)}")

    ## Split the data into train, validation and test
    data_train, data_val, data_test = database_split(words_list)
    print(f"Total training samples: {len(data_train)}")
    print(f"Total validation samples: {len(data_val)}")
    print(f"Total test samples: {len(data_test)}")

    ## Get the image paths and samples
    train_img_paths, train_labels = get_image_paths_and_labels(data_train)
    validation_img_paths, validation_labels = get_image_paths_and_labels(data_val)
    test_img_paths, test_labels = get_image_paths_and_labels(data_test)

    # Find maximum length and the size of the vocabulary in the training data.
    train_labels_clean = []
    CHARACTERS = set()
    MAX_LEN = 0

    for label in train_labels:
        label = label.split(" ")[-1].strip()
        for char in label:
            CHARACTERS.add(char)
        train_labels_clean.append(label)

    MAX_LEN = max(MAX_LEN, len(label))
    CHARACTERS = sorted(list(CHARACTERS))

    ## Clean the test and validation labels
    val_labels_clean = cleaning_labels(validation_labels)
    test_labels_clean = cleaning_labels(validation_labels)

    # Mapping characters to integers.
    char_to_num = StringLookup(vocabulary=list(CHARACTERS), mask_token=None)

    # Mapping integers back to original characters.
    num_to_char = StringLookup(
        vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    )

    AUTOTUNE = tensorflow.data.AUTOTUNE
    breakpoint()
    train_ds = prepare_dataset(train_img_paths, train_labels_clean)
    validation_ds = prepare_dataset(validation_img_paths, val_labels_clean)
    test_ds = prepare_dataset(test_img_paths, test_labels_clean)

    train_model(train_ds, validation_ds, test_ds)
