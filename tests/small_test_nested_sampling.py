from best import NestedSampler
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

def log_prob(x):
    x1 = x + tf.constant([-2.5, -2.5, 2.5], tf.float32)
    x2 = x + tf.constant([2.5, -2.5, 0], tf.float32)
    x3 = x + tf.constant([0., 2.5, -2.5], tf.float32)
    logL1 = -0.5 * tf.reduce_sum(x1 * x1, axis=-1)
    logL2 = -0.5 * tf.reduce_sum(x2 * x2, axis=-1)
    logL3 = -0.5 * tf.reduce_sum(x3 * x3, axis=-1)*0.5
    return tf.reduce_logsumexp(
        tf.concat([logL1[None, ...], logL2[None, ...], logL3[None, ...]], axis=0),
        axis=0
    )

d = 3

ns = NestedSampler(
    log_prob,
    ([-5]*d, [5]*d),
    1000,
    n_live_updates=10
)
    
results = ns.run(verbose=True, update_interval=10, display_param_idx=0)

print('Target logZ:',
      np.log(
          np.exp(np.log(2) + d/2*np.log(2*np.pi)-d*np.log(10)) +
          np.exp(d/2*np.log(2*np.pi/0.5)-d*np.log(10))
      )
)
