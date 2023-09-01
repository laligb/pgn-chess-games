import os
import time
from google.cloud import storage
from tensorflow import keras
import tensorflow


def convert_tflite(prediction_model, path):
    converter = tensorflow.lite.TFLiteConverter.from_keras_model(prediction_model)
    converter.optimizations = [tensorflow.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tensorflow.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tensorflow.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
    ]
    tf_lite_model = converter.convert()
    open(path, "wb").write(tf_lite_model)


def save_model(prediction_model) -> None:
    BUCKET_NAME = os.environ.get("BUCKET_NAME")
    LOCAL_DATA_PATH = os.environ.get("LOCAL_DATA_PATH")
    timestamp = time.strftime("%Y%m%d-%H%M")

    # Save model locally
    model_path = os.path.join(LOCAL_DATA_PATH, "models", f"{timestamp}.tflite")
    convert_tflite(prediction_model, model_path)

    print("✅ Model saved locally")

    model_filename = model_path.split("/")[-1]
    # e.g. "20230208-161047.tflite" for instance

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"models/{model_filename}")
    blob.upload_from_filename(model_path)

    print("✅ Model saved to GCS")

    return None


def load_interpreter():
    BUCKET_NAME = os.environ.get("BUCKET_NAME")
    LOCAL_DATA_PATH = os.environ.get("LOCAL_DATA_PATH")
    client = storage.Client()
    breakpoint()

    blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="models"))

    try:
        latest_blob = max(blobs, key=lambda x: x.updated)
        latest_model_path_to_save = os.path.join(LOCAL_DATA_PATH, latest_blob.name)
        latest_blob.download_to_filename(latest_model_path_to_save)

        interpreter = tensorflow.lite.Interpreter(latest_blob)
        interpreter.allocate_tensors()

        print("✅ Latest model downloaded from cloud storage")

        return interpreter
    except:
        print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

        return None
