import os
import time
from google.cloud import storage
from tensorflow import keras


def save_model(model: keras.Model = None) -> None:
    BUCKET_NAME = os.environ.get("BUCKET_NAME")
    LOCAL_DATA_PATH = os.environ.get("LOCAL_DATA_PATH")
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(LOCAL_DATA_PATH, "models", f"{timestamp}.h5")
    model.save(model_path)

    print("✅ Model saved locally")

    model_filename = model_path.split("/")[-1]  # e.g. "20230208-161047.h5" for instance
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"models/{model_filename}")
    blob.upload_from_filename(model_path)

    print("✅ Model saved to GCS")

    return None


def load_model():
    BUCKET_NAME = os.environ.get("BUCKET_NAME")
    LOCAL_DATA_PATH = os.environ.get("LOCAL_DATA_PATH")
    client = storage.Client()
    blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="model"))

    try:
        latest_blob = max(blobs, key=lambda x: x.updated)
        latest_model_path_to_save = os.path.join(LOCAL_DATA_PATH, latest_blob.name)
        latest_blob.download_to_filename(latest_model_path_to_save)

        latest_model = keras.models.load_model(latest_model_path_to_save)

        print("✅ Latest model downloaded from cloud storage")

        return latest_model
    except:
        print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

        return None
