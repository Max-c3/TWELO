from kestrix.params import *
from kestrix.model import create_new_model
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
    if model_name:
        print("Loading existing model.")
        model = keras.saving.load_model(f"../models/{model_name}")
    else:
        model = create_new_model()
    return model

def save_model(model):
    now = datetime.datetime.now().isoformat()
    model.save(f"../models/{now}.keras")
