import os
import pickle as pkl

import tensorflow as tf
import numpy as np

from best.client_emulators.custom_objects import Alsing, CustomTanh, create_msre_loss
from best.client_emulators import lcdm, sterile_neutrino

model_path = lcdm.path().joinpath("trained_model.keras")

def load_model_and_scalers(emulator_dir, root='best/client_emulators'):
    emulator_path = os.path.join(root, emulator_dir)
    if emulator_path == "best/client_emulators/lcdm":
        model_path = lcdm.path().joinpath("trained_model.keras")
        x_scaler_path = lcdm.path().joinpath("x_scaler.pkl")
        y_scaler_path = lcdm.path().joinpath("y_scaler.pkl")
    elif emulator_path == "best/client_emulators/sterile_neutrino":
        model_path = sterile_neutrino.path().joinpath("trained_model.keras")
        x_scaler_path = sterile_neutrino.path().joinpath("x_scaler.pkl")
        y_scaler_path = sterile_neutrino.path().joinpath("y_scaler.pkl")
    else:
        model_path = os.path.join(emulator_path, 'trained_model.keras')
        x_scaler_path = os.path.join(emulator_path, 'x_scaler.pkl')
        y_scaler_path = os.path.join(emulator_path, 'y_scaler.pkl')

    with open(x_scaler_path, 'rb') as f:
        x_scaler = pkl.load(f)
    with open(y_scaler_path, 'rb') as f:
        y_scaler = pkl.load(f)

    x_mean = tf.constant(x_scaler.mean_, dtype=tf.float32)
    x_scale = tf.constant(x_scaler.scale_, dtype=tf.float32)
    y_mean = tf.constant(y_scaler.mean_, dtype=tf.float32)
    y_scale = tf.constant(y_scaler.scale_, dtype=tf.float32)

    mean_square_relative_error = create_msre_loss(y_global_max=10.0, kappa=3.0, n=27.0, y_std=y_scale)
    custom_objects={'Alsing': Alsing, 'CustomTanh': CustomTanh, 'mean_square_relative_error': mean_square_relative_error}

    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    ## define box function that is exponentially increasing outside limits. It should be auto-differentiable.
    lower = tf.constant([
        0, 0, 0, 0, 0, 0.004, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9
    ], dtype=tf.float32)

    upper = tf.constant([
        3, 0.5, 2, 5, 2, 1, 0.8, 3.0, 200, 1, 10, 400, 400, 400, 400, 10, 50, 50, 100, 400, 10, 10, 10, 10, 10, 10, 3000, 3000, 1.1
    ], dtype=tf.float32)

    # Box is defined for the sterile neutrino model. If LCDM is loaded instead, two dimensions should be removed
    if 'lcdm' in emulator_dir:
        # remove indices 6 and 7 from lower and upper bounds for LCDM model
        lower = tf.concat([lower[:6], lower[8:]], axis=0)
        upper = tf.concat([upper[:6], upper[8:]], axis=0)

    @tf.function(reduce_retracing=True)
    def log_prob_fn(x):
        inp = (x - x_mean) / x_scale
        y_scaled = model(inp)
        log_like = tf.reshape(y_scaled * y_scale + y_mean, [-1])
        return log_like
    
    return log_prob_fn, lower, upper
