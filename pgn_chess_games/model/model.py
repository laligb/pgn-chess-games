import os

import tensorflow
from tensorflow import keras

# from google.cloud import storage


class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tensorflow.cast(tensorflow.shape(y_true)[0], dtype="int64")
        input_length = tensorflow.cast(tensorflow.shape(y_pred)[1], dtype="int64")
        label_length = tensorflow.cast(tensorflow.shape(y_true)[1], dtype="int64")

        input_length = input_length * tensorflow.ones(
            shape=(batch_len, 1), dtype="int64"
        )
        label_length = label_length * tensorflow.ones(
            shape=(batch_len, 1), dtype="int64"
        )
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions.
        return y_pred


def initialize_model(img_size):
    img_width = img_size[0]
    img_height = img_size[1]
    input_img = keras.Input(shape=(img_width, img_height, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))

    ## First CNN block
    x = keras.layers.Conv2D(
        32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(input_img)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)

    ## Second CNN layer
    x = keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(64, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)

    # RNNs.
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(128, return_sequences=True, dropout=0.25)
    )(x)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(64, return_sequences=True, dropout=0.25)
    )(x)

    # +2 is to account for the two special tokens introduced by the CTC loss.
    # The recommendation comes here: https://git.io/J0eXP.
    x = keras.layers.Dense(79 + 2, activation="softmax", name="dense2")(x)

    # Add CTC layer for calculating CTC loss at each step.
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model.
    model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="handwriting_recognizer"
    )
    # Optimizer.
    opt = keras.optimizers.Adam()
    # Compile the model and return.
    model.compile(optimizer=opt)
    return model
