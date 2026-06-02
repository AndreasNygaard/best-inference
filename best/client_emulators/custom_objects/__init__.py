import tensorflow as tf
import numpy as np

class CustomTanh(tf.keras.layers.Layer):
    def __init__(self, initial_alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.initial_alpha = initial_alpha

    def build(self, input_shape):
        self.alpha = self.add_weight(
            name='alpha',
            shape=(1,),
            initializer=tf.keras.initializers.Constant(self.initial_alpha),
            trainable=True
        )
        super().build(input_shape)

    @tf.function(jit_compile=True)
    def call(self, inputs):
        return tf.math.tanh(self.alpha * inputs)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "initial_alpha": self.initial_alpha
        })
        return config
    
class Alsing(tf.keras.layers.Layer):
    def __init__(self, initial_beta=1.0, initial_gamma=0.0, **kwargs):
        super().__init__(**kwargs)
        self.initial_beta = initial_beta
        self.initial_gamma = initial_gamma

    def build(self, input_shape):
        self.beta = self.add_weight(
            name="beta",
            shape=(1,),
            initializer=tf.keras.initializers.Constant(self.initial_beta),
            trainable=True
        )
        self.gamma = self.add_weight(
            name="gamma",
            shape=(1,),
            initializer=tf.keras.initializers.Constant(self.initial_gamma),
            trainable=True
        )
        super().build(input_shape)

    @tf.function(jit_compile=True)
    def call(self, inputs):
        return (self.gamma + (1 - self.gamma) / (1 + tf.exp(-self.beta * inputs))) * inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "initial_beta": self.initial_beta,
            "initial_gamma": self.initial_gamma
        })
        return config

def delta_chi2_from_k(kappa, n):
    kappa = tf.cast(kappa, tf.float32)
    n = tf.cast(n, tf.float32)

    mu = 1.0 - 2.0 / (9.0 * n)
    sigma = tf.sqrt(2.0 / (9.0 * n))
    return n * tf.pow(mu + kappa * sigma, 3.0)

def create_msre_loss(y_global_max, kappa, n, y_std):
    delta_chi2_k = delta_chi2_from_k(kappa, n)
    delta = -y_global_max - 0.5 * (delta_chi2_k / y_std)
    
    def mean_square_relative_error(y_true, y_pred):
        denominator = y_true + delta
        relative_error = (y_pred - y_true) / denominator
        return tf.reduce_mean(tf.square(relative_error))
    
    return mean_square_relative_error


def get_loss_function(loss_name, y_global_max=None, kappa=None, n=None, y_std=None):
    if loss_name == 'msre':
        return create_msre_loss(y_global_max, kappa, n, y_std)
    if loss_name in ['mse', 'mae']:
        return loss_name
    return loss_name
