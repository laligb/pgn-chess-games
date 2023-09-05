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

from pgn_chess_games.model.data import (
    get_image_paths_and_labels,
    get_constants,
    cleaning_labels,
    distortion_free_resize,
    prepare_dataset,
    prepare_predict_ds,
)
from pgn_chess_games.model.properties import model_properties
from pgn_chess_games.model.registry import (
    save_dictionary,
    save_num_char_dict,
    save_model,
    load_dictionary,
)
from pgn_chess_games.model.model import (
    initialize_model,
    decode_batch_predictions,
    get_predictions,
)
from pgn_chess_games.model.callback import EditDistanceCallback
from pgn_chess_games.model.predict_chess import box_extraction

epochs = 50


def run_model_chess():
    base_path = os.path.join(os.environ["LOCAL_DATA_PATH"], "dataset", "extracted")
    base_image_path = os.path.join(base_path)

    np.random.seed(42)
    tf.random.set_seed(42)

    train_words = open(f"{base_path}/train_data.txt", "r").readlines()
    train_samples = []
    for line in train_words:
        if line[0] == "#":
            continue
        if line.split(" ")[1] != "err":  # We don't need to deal with errored entries.
            train_samples.append(line)

    val_words = open(f"{base_path}/val_data.txt", "r").readlines()
    validation_samples = []
    for line in val_words:
        if line[0] == "#":
            continue
        if line.split(" ")[1] != "err":  # We don't need to deal with errored entries.
            validation_samples.append(line)

    test_words = open(f"{base_path}/testing_tags.txt", "r").readlines()
    test_samples = []
    for line in test_words:
        if line[0] == "#":
            continue
        if line.split(" ")[1] != "err":  # We don't need to deal with errored entries.
            test_samples.append(line)
    test_samples = validation_samples

    print(f"Total training samples: {len(train_samples)}")
    print(f"Total validation samples: {len(validation_samples)}")
    print(f"Total test samples: {len(test_samples)}")

    ## Get the image paths and samples
    breakpoint()
    train_img_paths, train_labels = get_image_paths_and_labels(train_samples)
    validation_img_paths, validation_labels = get_image_paths_and_labels(
        validation_samples
    )

    ## Clean the test and validation labels
    train_labels_cleaned, model_properties_dict = get_constants(train_labels)
    validation_labels_cleaned = cleaning_labels(validation_labels)

    AUTOTUNE = tensorflow.data.AUTOTUNE

    save_dictionary(model_properties_dict, "model_properties")

    characters = model_properties_dict["characters"]
    max_len = model_properties_dict["max_len"]

    # Mapping characters to integers.
    char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)

    # Mapping integers back to original characters.
    num_to_char = tensorflow.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    )

    num_to_char_dict, char_to_num_dict = save_num_char_dict(model_properties.characters)

    save_dictionary(num_to_char_dict, "num_to_char_dict")
    save_dictionary(char_to_num_dict, "char_to_num_dict")

    img_size = (128, 32)

    train_ds = prepare_dataset(train_img_paths, train_labels_cleaned)
    validation_ds = prepare_dataset(validation_img_paths, validation_labels_cleaned)

    print(f"⏳ Initializing model")
    model, prediction_model = initialize_model(img_size)
    edit_distance_callback = EditDistanceCallback(prediction_model, validation_ds)
    print(f"✅ Model initialized")

    # Train the model.
    print(f"⏳ Training model with {epochs} epochs")

    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=epochs,
        callbacks=[edit_distance_callback],
    )

    save_model(prediction_model, chess=True)


def predict_chess(cropped_dir_path):
    model_properties_dict = load_dictionary("model_properties")
    model_properties.get_constants(
        model_properties_dict["max_len"], model_properties_dict["characters"]
    )

    # Mapping characters to integers.
    char_to_num = StringLookup(
        vocabulary=list(model_properties.characters), mask_token=None
    )

    # Mapping integers back to original characters.
    num_to_char = StringLookup(
        vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    )

    pred_ds = prepare_predict_ds(cropped_dir_path)

    for batch in pred_ds:
        batch_images = batch["image"]
        predictions = get_predictions(batch_images)
        texts = decode_batch_predictions(predictions)
        print(texts)
    return texts


if __name__ == "__main__":
    LOCAL_DATA_PATH = os.environ["LOCAL_DATA_PATH"]
    PRED_PATH = os.path.join(LOCAL_DATA_PATH, "prediction/")
    img_for_box_extraction_path = (
        "/root/code/laligb/pgn-chess-games/data/data/017_0.png"
    )
    box_extraction(img_for_box_extraction_path, PRED_PATH)

    predict_chess(PRED_PATH)
