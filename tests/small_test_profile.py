import best
import tensorflow as tf

def log_prob(x):
    return -0.5 * tf.reduce_sum(x**2, axis=-1)

# --- correlated Gaussian precision ---
cov = tf.constant([
    [1.0, 0.4, 0.2],
    [0.4, 1.0, 0.4],
    [0.2, 0.4, 1.0]
], dtype=tf.float32)
inv_cov = tf.linalg.inv(cov)
log_det_cov = tf.linalg.logdet(cov)

def log_prob_corr(x):
    # x shape: (n_chains, 2)
    quad = tf.einsum("...i,ij,...j->...", x, inv_cov, x)
    return -0.5 * (quad + log_det_cov)

optimiser = best.Optimiser(log_prob, bounds=([-10, -10, -10], [10, 10, 10]))

results = optimiser.compute_profile(
    idxs=[0, 1]
)

optimiser = best.Optimiser(log_prob_corr, bounds=([-10, -10, -10], [10, 10, 10]))

results_corr = optimiser.compute_profile(
    idxs=[0, 1],
)

