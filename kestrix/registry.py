from kestrix.params import *
import datetime
import tensorflow as tf
from tensorflow import keras

def load_model(model_path=None):
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
    model = keras.saving.load_model(model_path)
    return model

def save_model(model):
    now = datetime.datetime.now()
    model.save(f"model_{now}.keras")
