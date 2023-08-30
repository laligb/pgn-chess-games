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
from pgn_chess_games.model.properties import model_properties
from pgn_chess_games.model.registry import save_model


LOCAL_DATA_PATH = os.path.join(os.environ["LOCAL_DATA_PATH"], "words")


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
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :21
    ]

    # Iterate over the results and get back the text.
    char_to_num = StringLookup(
        vocabulary=list(model_properties.characters), mask_token=None
    )
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
    ## Define the data with shuffle function
    np.random.seed(42)
    tensorflow.random.set_seed(42)
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

    ## Clean the test and validation labels
    train_labels_clean = get_constants(train_labels)
    val_labels_clean = cleaning_labels(validation_labels)
    test_labels_clean = cleaning_labels(validation_labels)

    AUTOTUNE = tensorflow.data.AUTOTUNE

    # Mapping characters to integers.
    char_to_num = StringLookup(
        vocabulary=list(model_properties.characters), mask_token=None
    )

    # Mapping integers back to original characters.
    num_to_char = StringLookup(
        vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    )

    img_size = (128, 32)

    train_ds = prepare_dataset(train_img_paths, train_labels_clean)
    validation_ds = prepare_dataset(validation_img_paths, val_labels_clean)
    test_ds = prepare_dataset(test_img_paths, test_labels_clean)

    print(f"⏳ Initializing model")
    model = initialize_model(img_size)
    print(f"✅ Model initialized")

    prediction_model = keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(name="dense2").output
    )
    edit_distance_callback = EditDistanceCallback(prediction_model, validation_ds)

    # Train the model.
    epochs = 1
    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=epochs,
        callbacks=[edit_distance_callback],
    )

    save_model(model=model)
