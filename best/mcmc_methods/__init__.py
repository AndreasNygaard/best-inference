import warnings

import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions
tf.get_logger().setLevel('ERROR')

from best.tools import LogProbCounter, py_update, trace_fn_w_progress_bar, trace_fn_wo_progress_bar, mh_proposal_fn, MalaWithStepSize, MalaResults, jit_tfp_sample


def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return "    Warning: " + str(msg) + '\n'
warnings.formatwarning = custom_formatwarning


### Metropolis-Hastings (MH) ###

def run_mh(log_prob_fn,
           initial_state,
           n_steps=1000,
           covmat=None,
           step_size=0.1,
           num_adaptation_steps=None,
           num_burnin_steps=0,
           n_chains=10,
           use_diagonal_covmat=False,
           progress_bar=True,
           jit_compile=True):

    if isinstance(initial_state, tf.Tensor):
        if len(initial_state.shape) == 1:
            initial_state = tf.repeat(tf.expand_dims(initial_state, axis=0), repeats=n_chains, axis=0)
        elif len(initial_state.shape) > 2:
            raise ValueError("If initial_state is a tensor, it must have shape (n_chains, n_params) or (n_params,).")
    else:
        raise ValueError("initial_state must be a tensor of shape (n_chains, n_params) or (n_params,).")

    n_chains = initial_state.shape[0]

    if covmat is None:
        scales = tf.math.reduce_std(initial_state, axis=0)
        if tf.reduce_any(scales == 0):
            covmat = tf.eye(initial_state.shape[1], dtype=tf.float32)
            warnings.warn("If covmat is not provided, it will be estimated from the initial state. However, if any parameter has zero variance across the initial chains, an identity covariance matrix will be used instead, which may lead to suboptimal performance. Consider providing a covariance matrix or ensuring that the initial state has non-zero variance across all parameters to mitigate this issue.")
        else:
            covmat = tf.power(scales,2) * tf.eye(initial_state.shape[1], dtype=tf.float32)
    elif len(covmat.shape) < 2:
        covmat = tf.power(covmat,2) * tf.eye(initial_state.shape[1], dtype=tf.float32)
    elif len(covmat.shape) > 2:
        raise ValueError("Covariance matrix must be either 2D, 1D, or scalar")
    if use_diagonal_covmat:
        covmat = tf.linalg.diag(tf.linalg.diag_part(covmat))
    L = tf.linalg.cholesky(covmat)

    def log_prob_whitened(z):
        x = initial_state + tf.linalg.matvec(L, z)
        log_det = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))
        return log_prob_fn(x) + log_det

    log_prob_counter = LogProbCounter(log_prob_whitened)

    step_size = tf.Variable(step_size, dtype=tf.float32)

    def step_size_getter_fn(kernel_results):
        return step_size
    def step_size_setter_fn(kernel_results, new_step_size):
        step_size.assign(new_step_size)
        return kernel_results

    mh_kernel = tfp.mcmc.RandomWalkMetropolis(
        target_log_prob_fn=log_prob_counter,
        new_state_fn=lambda state, seed: mh_proposal_fn(state, seed, step_size=step_size)
    )

    adaptive_mh = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=mh_kernel,
        num_adaptation_steps=num_burnin_steps,
        target_accept_prob=0.3,
        step_size_getter_fn=step_size_getter_fn,
        step_size_setter_fn=step_size_setter_fn,
    )

    z0 = tf.zeros_like(initial_state, dtype=tf.float32)

    if progress_bar:
        trace_fn = trace_fn_w_progress_bar
    else:
        trace_fn = trace_fn_wo_progress_bar

    samples, acceptance_rate = jit_tfp_sample(n_steps, num_burnin_steps, z0, adaptive_mh, progress_bar=progress_bar, jit_compile=jit_compile)
    x_samples = initial_state + tf.linalg.matmul(samples, L, transpose_b=True)
    n_evals = log_prob_counter.num_calls

    return x_samples, acceptance_rate, n_evals



### Affine Invariant Ensemble Sampler (AIES) ###

def aies_sampling(log_prob, n_steps, current_state, args=(), num_burnin_steps=0, progressbar=True, jit_compile=True):
    state1, state2 = current_state
    n_walkers, n_params = state1.shape

    state1 = tf.convert_to_tensor(state1)
    state2 = tf.convert_to_tensor(state2)

    logp1 = log_prob(state1)
    logp2 = log_prob(state2)

    dtype = state1.dtype
    n_params_m1 = tf.constant(n_params - 1.0, dtype=dtype)

    @tf.function(jit_compile=jit_compile)
    def run_chunk(state1, state2, logp1, logp2, steps):

        chain = tf.TensorArray(
            dtype=dtype,
            size=steps,
            element_shape=tf.TensorShape([2 * n_walkers, n_params]),
        )

        def body(i, state1, state2, logp1, logp2, chain):

            # --- same sequential update as before ---
            idx1 = tf.random.uniform([n_walkers], 0, n_walkers, dtype=tf.int32)
            partner1 = tf.gather(state2, idx1)

            z1 = 0.5 * (1.0 + tf.random.uniform([n_walkers], dtype=dtype)) ** 2
            z1r = tf.reshape(z1, [-1, 1])

            prop1 = partner1 + z1r * (state1 - partner1)
            logp_prop1 = log_prob(prop1)

            log_a1 = n_params_m1 * tf.math.log(z1) + (logp_prop1 - logp1)
            accept1 = tf.math.log(tf.random.uniform([n_walkers], dtype=dtype)) < log_a1

            accept1_f = tf.cast(accept1, dtype)[:, None]
            new_state1 = state1 * (1.0 - accept1_f) + prop1 * accept1_f
            new_logp1 = tf.where(accept1, logp_prop1, logp1)

            idx2 = tf.random.uniform([n_walkers], 0, n_walkers, dtype=tf.int32)
            partner2 = tf.gather(new_state1, idx2)

            z2 = 0.5 * (1.0 + tf.random.uniform([n_walkers], dtype=dtype)) ** 2
            z2r = tf.reshape(z2, [-1, 1])

            prop2 = partner2 + z2r * (state2 - partner2)
            logp_prop2 = log_prob(prop2)

            log_a2 = n_params_m1 * tf.math.log(z2) + (logp_prop2 - logp2)
            accept2 = tf.math.log(tf.random.uniform([n_walkers], dtype=dtype)) < log_a2

            accept2_f = tf.cast(accept2, dtype)[:, None]
            new_state2 = state2 * (1.0 - accept2_f) + prop2 * accept2_f
            new_logp2 = tf.where(accept2, logp_prop2, logp2)

            combined = tf.concat([new_state1, new_state2], axis=0)

            return (
                i + 1,
                new_state1,
                new_state2,
                new_logp1,
                new_logp2,
                chain.write(i, combined),
            )

        _, state1, state2, logp1, logp2, chain = tf.while_loop(
            lambda i, *_: i < steps,
            body,
            loop_vars=[0, state1, state2, logp1, logp2, chain],
            parallel_iterations=1,
        )

        return state1, state2, logp1, logp2, chain.stack()

    # -------- Python driver with progress bar --------
    total_steps = n_steps + num_burnin_steps
    chunk_size = min(max(1, total_steps // 100), 1000)
    chunk_remainder = total_steps % chunk_size

    # pre-compilation
    _ = run_chunk(
        state1, state2, logp1, logp2, chunk_size
    )
    if chunk_remainder > 0:
        _ = run_chunk(
            state1, state2, logp1, logp2, chunk_remainder
        )

    all_chunks = []
    steps_done = 0

    if progressbar:
        py_update(
            0,
            num_samples=n_steps,
            num_burnin_steps=num_burnin_steps,
            num_steps_between_results=0
        )

    while steps_done < total_steps:
        steps_this = min(chunk_size, total_steps - steps_done)

        state1, state2, logp1, logp2, chunk = run_chunk(
            state1, state2, logp1, logp2, steps_this
        )

        all_chunks.append(chunk)
        steps_done += steps_this

        if progressbar:
            py_update(
                steps_done,
                num_samples=n_steps,
                num_burnin_steps=num_burnin_steps,
                num_steps_between_results=0
            )

    samples = tf.concat(all_chunks, axis=0)
    samples = samples[num_burnin_steps:]
    return samples


def run_aies(log_prob_fn,
             initial_state,
             n_steps=1000,
             num_burnin_steps=0,
             progress_bar=True,
             jit_compile=True):

    if isinstance(initial_state, list):
        if len(initial_state) != 2:
            raise ValueError("If initial_state is a list, it must contain exactly two tensors.")
        if initial_state[0].shape != initial_state[1].shape:
            raise ValueError("If initial_state is a list, both tensors must have the same shape.")
        n_walkers = 2 * initial_state[0].shape[0]
    elif isinstance(initial_state, tf.Tensor):
        if len(initial_state.shape) != 2:
            raise ValueError("If initial_state is a tensor, it must have shape (n_walkers, n_params), where n_walkers is even.")
        elif initial_state.shape[0] % 2 != 0:
            raise ValueError("If initial_state is a tensor, the number of walkers (shape[0]) must be even.")
        n_walkers = initial_state.shape[0]
        initial_state = tf.split(initial_state, num_or_size_splits=2, axis=0)
    else:
        raise ValueError("initial_state must be either a tensor of shape (n_walkers, n_params) or a list of two tensors of shape (n_walkers/2, n_params), where n_walkers is even.")
    log_prob_counter = LogProbCounter(log_prob_fn)

    n_params = initial_state[0].shape[1]
    # run the sampler
    samples = aies_sampling(log_prob_counter,
                            n_steps,
                            initial_state,
                            args=[],
                            num_burnin_steps=num_burnin_steps,
                            progressbar=progress_bar,
                            jit_compile=jit_compile)
    acceptance_rate = tf.raw_ops.UniqueV2(x=samples, axis=[0])[0].shape[0]/samples.shape[0]
    n_evals = log_prob_counter.num_calls
    return samples, acceptance_rate, n_evals





### Hamiltonian Monte Carlo (HMC) ###

def run_hmc(log_prob_fn,
            initial_state,
            n_steps=100,
            covmat=None,
            num_leapfrog=5,
            step_size=0.1,
            num_adaptation_steps=None,
            num_burnin_steps=0,
            n_chains=10,
            use_diagonal_mass_matrix=False,
            progress_bar=True,
            jit_compile=True):

    if isinstance(initial_state, tf.Tensor):
        if len(initial_state.shape) == 1:
            initial_state = tf.repeat(tf.expand_dims(initial_state, axis=0), repeats=n_chains, axis=0)
        elif len(initial_state.shape) > 2:
            raise ValueError("If initial_state is a tensor, it must have shape (n_chains, n_params) or (n_params,).")
    else:
        raise ValueError("initial_state must be a tensor of shape (n_chains, n_params) or (n_params,).")

    n_chains = initial_state.shape[0]

    if covmat is None:
        scales = tf.math.reduce_std(initial_state, axis=0)
        if tf.reduce_any(scales == 0):
            covmat = tf.eye(initial_state.shape[1], dtype=tf.float32)
            warnings.warn("If covmat is not provided, it will be estimated from the initial state. However, if any parameter has zero variance across the initial chains, an identity covariance matrix will be used instead, which may lead to suboptimal performance. Consider providing a covariance matrix or ensuring that the initial state has non-zero variance across all parameters to mitigate this issue.")
        else:
            covmat = tf.power(scales,2) * tf.eye(initial_state.shape[1], dtype=tf.float32)
    elif len(covmat.shape) < 2:
        covmat = tf.power(covmat,2) * tf.eye(initial_state.shape[1], dtype=tf.float32)
    elif len(covmat.shape) > 2:
        raise ValueError("Covariance matrix must be either 2D, 1D, or scalar")
    if use_diagonal_mass_matrix:
        covmat = tf.linalg.diag(tf.linalg.diag_part(covmat))
    L = tf.linalg.cholesky(covmat)

    def log_prob_whitened(z):
        x = initial_state + tf.linalg.matvec(L, z)
        log_det = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))
        return log_prob_fn(x) + log_det

    log_prob_counter = LogProbCounter(log_prob_whitened)

    hcm_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=log_prob_counter,
        step_size=step_size,
        num_leapfrog_steps=num_leapfrog
    )
    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel = hcm_kernel,
        num_adaptation_steps=num_burnin_steps if num_adaptation_steps is None else num_adaptation_steps,
    )

    z0 = tf.zeros_like(initial_state, dtype=tf.float32)

    if progress_bar:
        trace_fn = trace_fn_w_progress_bar
    else:
        trace_fn = trace_fn_wo_progress_bar

    samples, acceptance_rate = jit_tfp_sample(n_steps, num_burnin_steps, z0, adaptive_hmc, progress_bar=progress_bar, jit_compile=jit_compile)
    x_samples = initial_state + tf.linalg.matmul(samples, L, transpose_b=True)
    n_evals = log_prob_counter.num_calls
    return x_samples, acceptance_rate, n_evals



### No-U-Turn Sampler (NUTS) ###

def run_nuts(log_prob_fn,
             initial_state,
             n_steps=100,
             covmat=None,
             max_tree_depth=6,
             step_size=0.1,
             num_adaptation_steps=None,
             target_accept_prob=0.8,
             num_burnin_steps=0,
             n_chains=10,
             use_diagonal_mass_matrix=False,
             progress_bar=True,
             jit_compile=True):

    if isinstance(initial_state, tf.Tensor):
        if len(initial_state.shape) == 1:
            initial_state = tf.repeat(tf.expand_dims(initial_state, axis=0), repeats=n_chains, axis=0)
        elif len(initial_state.shape) > 2:
            raise ValueError("If initial_state is a tensor, it must have shape (n_chains, n_params) or (n_params,).")
    else:
        raise ValueError("initial_state must be a tensor of shape (n_chains, n_params) or (n_params,).")

    n_chains = initial_state.shape[0]

    if covmat is None:
        scales = tf.math.reduce_std(initial_state, axis=0)
        if tf.reduce_any(scales == 0):
            covmat = tf.eye(initial_state.shape[1], dtype=tf.float32)
            warnings.warn("If covmat is not provided, it will be estimated from the initial state. However, if any parameter has zero variance across the initial chains, an identity covariance matrix will be used instead, which may lead to suboptimal performance. Consider providing a covariance matrix or ensuring that the initial state has non-zero variance across all parameters to mitigate this issue.")
        else:
            covmat = tf.power(scales,2) * tf.eye(initial_state.shape[1], dtype=tf.float32)
    elif len(covmat.shape) < 2:
        covmat = tf.power(covmat,2) * tf.eye(initial_state.shape[1], dtype=tf.float32)
    elif len(covmat.shape) > 2:
        raise ValueError("Covariance matrix must be either 2D, 1D, or scalar")
    if use_diagonal_mass_matrix:
        covmat = tf.linalg.diag(tf.linalg.diag_part(covmat))
    L = tf.linalg.cholesky(covmat)

    def log_prob_whitened(z):
        x = initial_state + tf.linalg.matvec(L, z)
        log_det = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))
        return log_prob_fn(x) + log_det

    log_prob_counter = LogProbCounter(log_prob_whitened)

    nuts = tfp.mcmc.NoUTurnSampler(
        target_log_prob_fn=log_prob_counter,
        step_size=step_size,
        max_tree_depth=max_tree_depth,
    )

    adaptive_nuts = tfp.mcmc.DualAveragingStepSizeAdaptation(
        nuts,
        num_adaptation_steps=num_burnin_steps if num_adaptation_steps is None else num_adaptation_steps,
        target_accept_prob=target_accept_prob
    )

    z0 = tf.zeros_like(initial_state, dtype=tf.float32)

    if progress_bar:
        trace_fn = trace_fn_w_progress_bar
    else:
        trace_fn = trace_fn_wo_progress_bar

    samples, acceptance_rate = jit_tfp_sample(n_steps, num_burnin_steps, z0, adaptive_nuts, progress_bar=progress_bar, jit_compile=jit_compile)
    x_samples = initial_state + tf.linalg.matmul(samples, L, transpose_b=True)
    n_evals = log_prob_counter.num_calls
    return x_samples, acceptance_rate, n_evals




### Metropolis Adjusted Langevin Algorithm (MALA) ###

def run_mala(log_prob_fn,
             initial_state,
             n_steps=100,
             covmat=None,
             step_size=0.01,
             num_burnin_steps=1000,
             num_steps_between_results=0,
             volatility_fn=None,
             n_chains=50,
             use_diagonal_covmat=False,
             progress_bar=True,
             jit_compile=True):

    if isinstance(initial_state, tf.Tensor):
        if len(initial_state.shape) == 1:
            initial_state = tf.repeat(tf.expand_dims(initial_state, axis=0), repeats=n_chains, axis=0)
        elif len(initial_state.shape) > 2:
            raise ValueError("If initial_state is a tensor, it must have shape (n_chains, n_params) or (n_params,).")
    else:
        raise ValueError("initial_state must be a tensor of shape (n_chains, n_params) or (n_params,).")

    n_chains = initial_state.shape[0]

    if covmat is None:
        scales = tf.math.reduce_std(initial_state, axis=0)
        if tf.reduce_any(scales == 0):
            covmat = tf.eye(initial_state.shape[1], dtype=tf.float32)
            warnings.warn("If covmat is not provided, it will be estimated from the initial state. However, if any parameter has zero variance across the initial chains, an identity covariance matrix will be used instead, which may lead to suboptimal performance. Consider providing a covariance matrix or ensuring that the initial state has non-zero variance across all parameters to mitigate this issue.")
        else:
            covmat = tf.power(scales,2) * tf.eye(initial_state.shape[1], dtype=tf.float32)
    elif len(covmat.shape) < 2:
        covmat = tf.power(covmat,2) * tf.eye(initial_state.shape[1], dtype=tf.float32)
    elif len(covmat.shape) > 2:
        raise ValueError("Covariance matrix must be either 2D, 1D, or scalar")
    if isinstance(step_size, tf.Tensor) and len(step_size.shape)==1 and step_size.shape[0]==covmat.shape[0]:
        covmat = covmat * tf.tensordot(step_size, step_size, axes=0)
        step_size=1.0
    if use_diagonal_covmat:
        covmat = tf.linalg.diag(tf.linalg.diag_part(covmat))

    L = tf.linalg.cholesky(covmat)

    def log_prob_whitened(z):
        x = initial_state + tf.linalg.matvec(L, z)
        log_det = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))
        return log_prob_fn(x) + log_det

    log_prob_counter = LogProbCounter(log_prob_whitened)

    if volatility_fn is None:
        def volatility_fn(x):
            return 1. / (0.5 + 0.1 * tf.math.abs(x))

    z0 = tf.zeros_like(initial_state, dtype=tf.float32)

    def mala_step_size_getter_fn(kernel_results):
        return kernel_results.step_size

    def mala_step_size_setter_fn(kernel_results, new_step_size):
        return MalaResults(
            inner_results=kernel_results.inner_results,
            step_size=new_step_size
        )

    wrapped_mala = MalaWithStepSize(
        target_log_prob_fn=log_prob_counter,
        step_size=step_size,
        volatility_fn=volatility_fn
    )
    adaptive_mala = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=wrapped_mala,
        num_adaptation_steps=num_burnin_steps,
        target_accept_prob=0.574,
        step_size_getter_fn=mala_step_size_getter_fn,
        step_size_setter_fn=mala_step_size_setter_fn,
    )

    if progress_bar:
        trace_fn = trace_fn_w_progress_bar
    else:
        trace_fn = trace_fn_wo_progress_bar

    samples, acceptance_rate = jit_tfp_sample(n_steps, num_burnin_steps, z0, adaptive_mala, progress_bar=progress_bar, inner_level=2, jit_compile=jit_compile)
    x_samples = initial_state + tf.linalg.matmul(samples, L, transpose_b=True)
    n_evals = log_prob_counter.num_calls.numpy()
    return x_samples, acceptance_rate, n_evals
