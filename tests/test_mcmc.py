import tensorflow as tf
import numpy as np

from best.client_emulators import load_model_and_scalers
from best import Sampler


if 'GPU' in [x.device_type for x in tf.config.list_physical_devices()]:
    device = '/GPU:0'
else:
    device = '/CPU:0'

with tf.device(device):

    log_prob_lcdm, lower_lcdm, upper_lcdm = load_model_and_scalers('lcdm')

    box_center = (lower_lcdm + upper_lcdm) / 2

    mu_planck = tf.constant(np.array([2.235, 0.1194, 1.042, 3.04, 0.965, 0.0546, 33.29, 0.3325, 5.377, 173.8, 57.1, 29.52, 127.1, 4.046, 5.413, 10.89, 24.27, 91.74, 0.01359, 0.142, 0.5247, 0.2598, 0.8805, 2.113, 997, 995.6, 0.9989]),dtype=tf.float32)

    box_center = mu_planck
    progressbar = True
    jit=True
    s = Sampler(log_prob_lcdm, bounds=(lower_lcdm, upper_lcdm))

    res_aies = s.sample(method='aies', n_steps=5000, n_chains=100, initial_state=box_center, initial_distribution='gaussian', num_burnin_steps=5000, num_covmat_updates=1, sampler_kwargs={'progress_bar':progressbar}, jit_compile=jit)
    
    res_hmc = s.sample(method='hmc', n_steps=5000, n_chains=100, initial_state=box_center, initial_distribution='repeat', num_burnin_steps=500, num_covmat_updates=3, sampler_kwargs={'num_burnin_steps': 500, 'progress_bar':progressbar}, jit_compile=jit, update_initial_state=True)
    
    res_nuts = s.sample(method='nuts', n_steps=5000, n_chains=100, initial_state=box_center, initial_distribution='repeat', num_burnin_steps=500, num_covmat_updates=3, sampler_kwargs={'num_burnin_steps': 500, 'progress_bar':progressbar}, jit_compile=jit, update_initial_state=True)

    res_mh = s.sample(method='mh', n_steps=5000, n_chains=100, initial_state=box_center, initial_distribution='repeat', num_burnin_steps=500, num_covmat_updates=3, sampler_kwargs={'num_burnin_steps': 500, 'progress_bar':progressbar}, jit_compile=jit, update_initial_state=True)

    res_mala = s.sample(method='mala', n_steps=5000, n_chains=100, initial_state=box_center, initial_distribution='repeat', num_burnin_steps=500, num_covmat_updates=5, sampler_kwargs={'num_burnin_steps': 500, 'progress_bar':progressbar}, jit_compile=jit, update_initial_state=True)

