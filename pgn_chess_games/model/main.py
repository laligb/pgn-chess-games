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
)
from pgn_chess_games.model.model import (
    initialize_model,
    initialize_pred_model,
    decode_batch_predictions,
)
from pgn_chess_games.model.callback import EditDistanceCallback
from pgn_chess_games.model.properties import model_properties
from pgn_chess_games.model.registry import save_model, load_model
from pgn_chess_games.utils import img_base64_to_num, preproc_image

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
    model = initialize_model(img_size)
    prediction_model = initialize_pred_model(model)
    edit_distance_callback = EditDistanceCallback(prediction_model, validation_ds)
    print(f"✅ Model initialized")

    # Train the model.
    epochs = 50
    print(f"⏳ Training model with {epochs} epochs")

    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=epochs,
        callbacks=[edit_distance_callback],
    )

    save_model(model=model)


# TODO align preprocessing with API
# def prediction(images):
#     img_array = img_base64_to_num(images)
#     img_boxes = preproc_image(img_array)
#     images_ds = tensorflow.data.Dataset.from_tensor_slices(img_boxes)

#     model = load_model()
#     preds = model.predict(images_ds)
#     pred_texts = decode_batch_predictions(preds)
#     return pred_texts
