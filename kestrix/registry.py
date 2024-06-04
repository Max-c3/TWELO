from kestrix.params import *
import datetime
import tensorflow as tf
from tensorflow import keras

def load_model(model_name=None):
    """Load a model via model_path.

    Parameters
    ----------
    model_path : str, optional
        _description_, by default None

    Returns
    -------
    Keras model
        _description_
    """
    print("Loading existing model.")
    model = keras.saving.load_model(f"models/{model_name}.keras")

    return model

def save_model(model, model_name=None):
    if not model_name:
        now = datetime.datetime.now().isoformat()
        model_name = f"{now}"
    print(f"Saving model {model_name}.")
    model.save(f"kaggle/working/{model_name}.keras")
    save_to_bucket(model_name)

def save_to_bucket(name):
    storage_client = storage.Client(project='le-wagon-419714')
    bucket = storage_client.get_bucket("kestrix")
    blob = bucket.blob(f"models/{name}.keras")
    blob.upload_from_filename(f"kaggle/working/{name}.keras")
    print('Model {} uploaded.'.format(
        name))
