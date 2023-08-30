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
