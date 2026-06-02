# BEST

(**B**atched **E**mulator **S**ampling with **T**ensorFlow)

A TensorFlow-based inference framework for high-performance Markov Chain Monte Carlo (MCMC) sampling, including support for neural likelihood emulators ("client emulators") and adaptive covariance estimation.

---

## Overview

`best` provides a unified interface for sampling using multiple MCMC algorithms with GPU acceleration via TensorFlow:

### Supported samplers

- Metropolis-Hastings (MH)
- Affine Invariant Ensemble Sampler (AIES)
- Hamiltonian Monte Carlo (HMC)
- No-U-Turn Sampler (NUTS)
- Metropolis Adjusted Langevin Algorithm (MALA)

### Key features

- TensorFlow / GPU acceleration
- Automatic covariance matrix adaptation
- Bounded parameter inference
- Parallelized multi-chain sampling
- Pretrained neural likelihood emulators
- JIT compilation (XLA support)

---

## Installation

### From PyPI

```bash
pip install best-inference
```
### From source

```bash
git clone https://github.com/AndreasNygaard/best-inference.git
cd best-inference
pip install .
```
---
## Quick start

```python
import best
import tensorflow as tf

def log_prob(x):
    return -0.5 * tf.reduce_sum(x**2, axis=-1)

sampler = best.Sampler(log_prob, bounds=([-5, -5], [5, 5]))

results = sampler.sample(
    method="hmc",
    n_steps=2000,
    n_chains=50,
    initial_distribution="uniform",
    num_burnin_steps=1000
)

print(results.samples.shape)
```
---
## Sampler API

### Initialisation

```python
best.Sampler(
    log_prob_fn,
    bounds=None,
    enforce_boundaries=True,
    covmat=None,
    initial_state=None,
    n_chains=10,
    initial_distribution="repeat"
)
```

### Sampling

```python
results = sampler.sample(
    method="mh" | "aies" | "hmc" | "nuts" | "mala",
    n_steps=1000,
    n_chains=10,
    initial_state=None,
    initial_distribution="repeat" | "uniform" | "gaussian",
    bounds=None,
    covmat=None,
    num_burnin_steps=100,
    num_covmat_updates=None,
    sampler_kwargs={},
    burnin_kwargs={},
    get_individual_chains=True,
    jit_compile=True
)
```

### Output

```python
results.samples
results.acceptance_rate
results.evaluations

results.burnin_samples
results.burnin_acceptance_rates
results.burnin_evaluations
results.covmat_estimate
```

## Client emulators

BEST includes pretrained neural likelihood emulators for cosmology-inspired inference problems.

### Available models

 - lcdm
 - sterile_neutrino

### Load a model

```python
from best.client_emulators import load_model_and_scalers

log_prob_fn, lower_bounds, upper_bounds = load_model_and_scalers("lcdm")
```

### Example: emulator-based inference

```
import best
from best.client_emulators import load_model_and_scalers

log_prob_fn, lower, upper = load_model_and_scalers("lcdm")

sampler = best.Sampler(log_prob_fn, bounds=(lower, upper))

results = sampler.sample(
    method="aies",
    n_steps=5000,
    n_chains=100,
    initial_distribution="uniform",
    num_burnin_steps=2000,
    num_covmat_updates=1
)
```

## Supported Algorithms
### Metropolis-Hastings (MH)
Random-walk MCMC with optional adaptive covariance scaling.
### Affine Invariant Ensemble Sampler (AIES)
Efficient for highly anisotropic or correlated parameter spaces.
### Hamiltonian Monte Carlo (HMC)
Gradient-based sampling with leapfrog integration.
### No-U-Turn Sampler (NUTS)
Adaptive HMC variant with automatic trajectory length selection.
### Metropolis Adjusted Langevin Algorithm (MALA)
Gradient-informed diffusion-based sampler.

## Performance Notes
 - TensorFlow enables GPU acceleration where available
 - JIT compilation (XLA) improves performance for large chains
 - Vectorized multi-chain execution is used throughout
 - Covariance estimation is performed during burn-in when enabled

## Example: Multi-sampler comparison

```python
sampler.set_initial_state(initial_state=means, covmat=covmat, initial_distribution="gaussian")
res_aies = sampler.sample(method="aies", n_steps=5000, n_chains=100)
res_hmc  = sampler.sample(method="hmc",  n_steps=5000, n_chains=100)
res_nuts = sampler.sample(method="nuts", n_steps=5000, n_chains=100)
res_mh   = sampler.sample(method="mh",   n_steps=5000, n_chains=100)
res_mala = sampler.sample(method="mala", n_steps=5000, n_chains=100)
```

## Requirements
 - Python ≥ 3.10
 - TensorFlow ≥ 2.17
 - TensorFlow Probability ≥ 0.24
 - NumPy
 - tf-keras
 - hypersphere-sampler

## Citation
If you use this package, please cite:

Citation will be added once the accompanying paper is available (arXiv preprint in preparation).

## Contributing
Contributions are welcome.
###Steps:
 - Fork repository
 - Create feature branch
 - Add tests in ```tests/```
 - Submit pull request

## License
#### MIT License
Copyright (c) 2026 Andreas Nygaard
