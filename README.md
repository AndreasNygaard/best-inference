# BEST

(**B**atched **E**mulator **S**ampling with **T**ensorFlow)

A TensorFlow-based inference framework for high-performance Markov Chain Monte Carlo (MCMC) sampling and profile likelihood optimisation, including support for neural likelihood emulators and adaptive covariance estimation.

---

## Overview

`best` provides a unified interface for sampling using multiple MCMC algorithms with GPU acceleration via TensorFlow. It also features an optimisation module for computing profile likelihoods with an arbitrary number of fixed parameters.

### Supported mcmc samplers

- Metropolis-Hastings (MH)
- Affine Invariant Ensemble Sampler (AIES)
- Hamiltonian Monte Carlo (HMC)
- No-U-Turn Sampler (NUTS)
- Metropolis Adjusted Langevin Algorithm (MALA)

### Key features

- TensorFlow / GPU acceleration
- Automatic covariance matrix adaptation
- Bounded parameter inference
- Parallelised multi-chain sampling
- parallelised optimisation of profile likelihoods
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

### Bayesian MCMC sampling 

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
print(results.loglkl.shape)
```

### Frequentist profile likelihoods

```python
import best
import tensorflow as tf

def log_prob(x):
    return -0.5 * tf.reduce_sum(x**2, axis=-1)

optimiser = best.Optimiser(log_prob, bounds=([-5, -5, -5], [5, 5, 5]))

# 2D profile with the first two parameters fixed (0 and 1)
results = optimiser.compute_profile([0,1])

print(results.full_position.shape)
print(results.loglkl.shape)
```

---
## Sampler API

### Initialisation

```python
sampler = best.Sampler(
    log_prob_fn,
    bounds=None,
    enforce_boundaries=True,
    covmat=None,
    initial_state=None,
    n_chains=10,
    initial_distribution="repeat",
    boundary_penalty_factor=10000
)
```
```python
optimiser = best.Optimiser(
    log_prob_fn,
    bounds,
    covmat=None,
    loc=None,
    mcmc_temperature=1.0
)
```

### Sampling

```python
results_samp = sampler.sample(
    method="mh" | "aies" | "hmc" | "nuts" | "mala",
    n_steps=1000,
    n_chains=10,
    initial_state=None,
    initial_distribution="repeat" | "uniform" | "gaussian",
    bounds=None,
    covmat=None,
    num_burnin_steps=100,
    num_covmat_updates=None,
    update_initial_state=True,
    update_initial_distribution=True,
    continue_distribution=False,
    sampler_kwargs={},
    burnin_kwargs={},
    get_individual_chains=True,
    jit_compile=True,
    temperature=1.0
)
```

### Optimisation

```python
results_opt = optimiser.compute_profile(
    idxs=[], # indices for fixed parameters
    fixed_points=None,
    nbins=20,
    batch_size=10,
    start_temperature=1.0,
    decay_temperature=0.5,
    min_temperature=1e-2,
    step_size=0.05,
    min_step_size=1e-5,
    decay_step_size=0.5,
    max_correct_loglike=10000,
    nd_fixed=None,
    verbose=True
)
```

### Output

```python
results_samp.samples
results_samp.loglkl
results_samp.acceptance_rate
results_samp.evaluations

results_samp.burnin_samples
results_samp.burnin_loglkl
results_samp.burnin_acceptance_rates
results_samp.burnin_evaluations
results_samp.covmat_estimate
```
```python
results_opt.fixed_points
results_opt.loglkl
results_opt.reduced_position
results_opt.full_position
results_opt.idxs
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

### Example: emulator-based profile likelihood

```
import best
from best.client_emulators import load_model_and_scalers

log_prob_fn, lower, upper = load_model_and_scalers("lcdm")

optimiser = best.Optimiser(log_prob_fn, bounds=(lower, upper))

# 2D profile for omega_b and omega_cdm
results = optimiser.compute_profile(
    idxs=[0,1]
)
```

## Supported MCMC algorithms
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

## Performance notes
 - TensorFlow enables GPU acceleration where available
 - JIT compilation (XLA) improves performance for large chains
 - Vectorized multi-chain execution is used throughout
 - Covariance estimation is performed during burn-in when enabled
 - Optimiser for profile likelihoods is initialised with an MCMC for exploring the parameter space

## Example: Multi-sampler comparison

```python
sampler.set_initial_state(initial_state=means, covmat=covmat, initial_distribution="gaussian")
res_aies = sampler.sample(method="aies", n_steps=5000, n_chains=100)
res_hmc  = sampler.sample(method="hmc",  n_steps=5000, n_chains=100)
res_nuts = sampler.sample(method="nuts", n_steps=5000, n_chains=100)
res_mh   = sampler.sample(method="mh",   n_steps=5000, n_chains=100)
res_mala = sampler.sample(method="mala", n_steps=5000, n_chains=100)
```

## Refining profile likelihoods
The optimiser is initialised by running an MCMC sampler in order to explore the parameter space and estimate the covariance matrix and the best-fit point. The points sampled here allow for an automatic selection of relevant points for the 1D and 2D profile likelihoods (as to not waste computational effort on bad points in a grid).

It can, however, happen that a few points fail to optimise properly, and this can be inspected using the `plot_profile_1d` and `plot_profile_2d` methods producing plots like these (with 1-sigma, 2-sigma, and 3-sigma contours shown as well):

```python
results = optimiser.compute_profile([0,1])
optimiser.plot_profile_2d(results)
```
<img width="449" height="382" alt="plot_profile" src="https://raw.githubusercontent.com/AndreasNygaard/best-inference/main/assets/plot_profile.png" />

Here, there are three points that stand out (artificially altered for this example), and these can be recomputed using the methods `recompute_points_1d` and `recompute_points_2d`. This will open an interactive version of the plot where points can be selected by clicking them and recomputed using the "Enter" key:
 
```python
updated_results = optimiser.recompute_points_2d(results)
```
<img width="446" height="382" alt="recompute" src="https://raw.githubusercontent.com/AndreasNygaard/best-inference/main/assets/recompute.gif" />

Even though the automatic point selection worked very well, sometimes a few more points are needed to properly represent the 3-sigma contour well enough. In this case, one can use the methods `add_points_1d` and `add_points_2d`. This will also open an interactive version of the plot where new points can be added by clicking the desired position and computed using the "Enter" key: 

```python
updated_results = optimiser.recompute_points_2d(updated_results)
```

<img width="446" height="382" alt="add" src="https://raw.githubusercontent.com/AndreasNygaard/best-inference/main/assets/add.gif" />

When adding or recomputing points for a 2D profile likelihood, the colour scale can be adjusted using the "up" and "down" arrow keys. This can help better compare adjacent points when the span in likelihood values is quite large:

<img width="446" height="382" alt="color_scale" src="https://raw.githubusercontent.com/AndreasNygaard/best-inference/main/assets/color_scale.gif" />

## Requirements
 - Python ≥ 3.10
 - TensorFlow ≥ 2.17
 - TensorFlow Probability ≥ 0.24
 - NumPy
 - tf-keras
 - hypersphere-sampler

## Citation
If you use this package, please cite:

```
@article{Nygaard:2026fgl,
    author = "Nygaard, Andreas and Janken, Luca and Hannestad, Steen and Tram, Thomas",
    title = "{Posterior sampling in the Age of Emulators}",
    eprint = "2606.04895",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.IM",
    month = "6",
    year = "2026"
}
```

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