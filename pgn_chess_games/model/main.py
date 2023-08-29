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
LOCAL_DATA_PATH = os.path.join(os.environ["LOCAL_DATA_PATH"], "words")


def train_model(train_ds, validation_ds, test_ds, epochs=10):
    size = (128, 32)
    model = initialize_model(size)
    prediction_model = keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(name="dense2").output
    )
    edit_distance_callback = EditDistanceCallback(prediction_model)

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


def main():
    AUTOTUNE = tensorflow.data.AUTOTUNE
    words_list = define_data(LOCAL_DATA_PATH)
    data_train, data_val, data_test = database_split(words_list)
    breakpoint()
    train_ds, validation_ds, test_ds = image_processing(data_train, data_val, data_test)
    validation_images, validation_labels = callback_indices(validation_ds)
    train_model(train_ds, validation_ds, test_ds)
