from best import NestedSampler
import tensorflow as tf
import numpy as np

from best.client_emulators import load_model_and_scalers
log_prob_fn, lower, upper = load_model_and_scalers("lcdm")

if 'GPU' in [x.device_type for x in tf.config.list_physical_devices()]:
    device = '/GPU:0'
    print('Using GPU')
else:
    device = '/CPU:0'

with tf.device(device):
    ns = NestedSampler(
        log_prob_fn,
        (lower,upper)
        10000,
        n_live_updates=1000,
        n_max_iter=10000
    )
    
    results = ns.run(verbose=True, update_interval=10, display_param_idx=0)
