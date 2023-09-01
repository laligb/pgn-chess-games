import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow
from tensorflow.keras.layers import StringLookup
from pgn_chess_games.model.data import (
    define_data,
    database_split,
    get_image_paths_and_labels,
    get_constants,
    cleaning_labels,
    prepare_dataset,
    prepare_prediction_dataset,
    preproc_predictions,
)
from pgn_chess_games.model.model import (
    initialize_model,
    decode_batch_predictions,
)
from pgn_chess_games.model.callback import EditDistanceCallback
from pgn_chess_games.model.properties import model_properties
from pgn_chess_games.model.registry import save_model, load_interpreter

LOCAL_DATA_PATH = os.path.join(os.environ["LOCAL_DATA_PATH"], "words")


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

    ## Clean the test and validation labels
    train_labels_clean = get_constants(train_labels)
    val_labels_clean = cleaning_labels(validation_labels)

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

    print(f"⏳ Initializing model")
    model, prediction_model = initialize_model(img_size)
    edit_distance_callback = EditDistanceCallback(prediction_model, validation_ds)
    print(f"✅ Model initialized")

    # Train the model.
    epochs = 1
    print(f"⏳ Training model with {epochs} epochs")

    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=epochs,
        callbacks=[edit_distance_callback],
    )

    save_model(prediction_model)


def get_predictions(input_batch):
    interpreter = load_interpreter()
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


def get_predictions_decoded(batch):
    reshaped_arrays = []
    for i in range(len(batch)):
        tmp_array = None
        tmp_array = np.expand_dims(batch[i], axis=-1)
        new_array = preproc_predictions(tmp_array)
        new_array = new_array / 255.0
        reshaped_arrays.append(new_array)

    batch_images = np.array(reshaped_arrays)

    input_batch = batch_images
    predictions = get_predictions(input_batch)
    pred_texts = decode_batch_predictions(predictions)

    return pred_texts
