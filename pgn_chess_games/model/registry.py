import os
import time
from google.cloud import storage
from tensorflow import keras
import tensorflow
import json


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

    blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="models"))
    try:
        latest_blob = max(blobs, key=lambda x: x.updated)
        latest_model_path_to_save = os.path.join(LOCAL_DATA_PATH, latest_blob.name)
        latest_blob.download_to_filename(latest_model_path_to_save)

        interpreter = tensorflow.lite.Interpreter(latest_model_path_to_save)

        print("✅ Latest model downloaded from cloud storage")
        return interpreter
    except:
        print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

        return None


def save_num_char_dict(characters):
    char_to_num_dict = dict()
    i = 0
    for char in characters:
        char_to_num_dict[char] = i
        i += 1
    num_to_char_dict = {value: key for key, value in char_to_num_dict.items()}

    return num_to_char_dict, char_to_num_dict


def save_dictionary(dictionary, name: str) -> None:
    BUCKET_NAME = os.environ.get("BUCKET_NAME")
    LOCAL_DATA_PATH = os.environ.get("LOCAL_DATA_PATH")

    # Save dictionary locally as .json file
    file = f"{name}.json"
    dictionary_path = os.path.join(LOCAL_DATA_PATH, "dictionary", file)

    with open(dictionary_path, "w") as fp:
        json.dump(dictionary, fp)

    print(f"✅ {name} dictionary saved locally")

    # e.g. "20230208-161047.tflite" for instance

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"dictionary/{file}")
    blob.upload_from_filename(dictionary_path)

    print(f"✅ {name} dictionary saved to GCS")

    return None
