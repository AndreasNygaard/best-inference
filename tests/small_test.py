import best
import tensorflow as tf

def log_prob(x):
    return -0.5 * tf.reduce_sum(x**2, axis=-1)

# --- correlated Gaussian precision ---
cov = tf.constant([
    [1.0, 0.8],
    [0.8, 1.0]
], dtype=tf.float32)
inv_cov = tf.linalg.inv(cov)
log_det_cov = tf.linalg.logdet(cov)

def log_prob_corr(x):
    # x shape: (n_chains, 2)
    quad = tf.einsum("...i,ij,...j->...", x, inv_cov, x)
    return -0.5 * (quad + log_det_cov)

sampler = best.Sampler(log_prob, bounds=([-10, -10], [10, 10]))

results = sampler.sample(
    method="hmc",
    n_steps=2000,
    n_chains=50,
    initial_distribution="uniform",
    num_burnin_steps=1000
)

sampler = best.Sampler(log_prob_corr, bounds=([-10, -10], [10, 10]))

results_corr = sampler.sample(
    method="hmc",
    n_steps=2000,
    n_chains=50,
    initial_distribution="uniform",
    num_burnin_steps=1000
)

