import numpy as np
import tensorflow


def calculate_edit_distance(labels, predictions, max_len=27):
    # Get a single batch and convert its labels to sparse tensors.
    saprse_labels = tensorflow.cast(
        tensorflow.sparse.from_dense(labels), dtype=tensorflow.int64
    )

    # Make predictions and convert them to sparse tensors.
    input_len = np.ones(predictions.shape[0]) * predictions.shape[1]
    predictions_decoded = tensorflow.keras.backend.ctc_decode(
        predictions, input_length=input_len, greedy=True
    )[0][0][:, :max_len]
    sparse_predictions = tensorflow.cast(
        tensorflow.sparse.from_dense(predictions_decoded), dtype=tensorflow.int64
    )

    # Compute individual edit distances and average them out.
    edit_distances = tensorflow.edit_distance(
        sparse_predictions, saprse_labels, normalize=False
    )
    return tensorflow.reduce_mean(edit_distances)


def callback_indices(validation_ds):
    ## Creating batches
    validation_images = []
    validation_labels = []
    for batch in validation_ds:
        validation_images.append(batch["image"])
        validation_labels.append(batch["label"])
    return validation_images, validation_labels


class EditDistanceCallback(tensorflow.keras.callbacks.Callback):
    def __init__(self, pred_model, validation_ds):
        super().__init__()
        self.prediction_model = pred_model
        self.validation_ds = validation_ds

    def on_epoch_end(self, epoch, logs=None):
        edit_distances = []
        validation_images, validation_labels = callback_indices(self.validation_ds)

        for i in range(len(validation_images)):
            labels = validation_labels[i]
            predictions = self.prediction_model.predict(validation_images[i])
            edit_distances.append(calculate_edit_distance(labels, predictions).numpy())

        print(
            f"Mean edit distance for epoch {epoch + 1}: {np.mean(edit_distances):.4f}"
        )
