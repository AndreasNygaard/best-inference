import tensorflow as tf
import tensorflow_probability as tfp
import time
import os
import sys
from collections import namedtuple

class LogProbCounter:
    def __init__(self, log_prob_fn):
        self.log_prob_fn = log_prob_fn
        self.num_calls = tf.Variable(0, dtype=tf.int64, trainable=False)

    def __call__(self, *args):
        self.num_calls.assign_add(1)
        return self.log_prob_fn(*args)


def jit_tfp_sample(n_steps, num_burnin_steps, current_state, kernel, progress_bar=True, inner_level=1, jit_compile=True):
    kernel_results = kernel.bootstrap_results(current_state)

    @tf.function(jit_compile=jit_compile)
    def run_chunk1(current_state, kernel_results, num_steps):
        states = tf.TensorArray(current_state.dtype, size=num_steps)
        accepts = tf.TensorArray(tf.bool, size=num_steps)
        loglkl = tf.TensorArray(current_state.dtype, size=num_steps)

        def body(i, state, results, states, accepts, loglkl):
            next_state, next_results = kernel.one_step(state, results)

            states = states.write(i, next_state)
            accepts = accepts.write(i, next_results.inner_results.is_accepted)
            loglkl = loglkl.write(i, next_results.inner_results.accepted_results.target_log_prob)

            return i + 1, next_state, next_results, states, accepts, loglkl

        _, state, results, states, accepts, loglkl = tf.while_loop(
            lambda i, *_: i < num_steps,
            loop_vars=[0, current_state, kernel_results, states, accepts, loglkl],
            body=body,
            parallel_iterations=1,
        )

        return state, results, states.stack(), accepts.stack(), loglkl.stack()

    @tf.function(jit_compile=jit_compile)
    def run_chunk2(current_state, kernel_results, num_steps):
        states = tf.TensorArray(current_state.dtype, size=num_steps)
        accepts = tf.TensorArray(tf.bool, size=num_steps)
        loglkl = tf.TensorArray(current_state.dtype, size=num_steps)

        def body(i, state, results, states, accepts, loglkl):
            next_state, next_results = kernel.one_step(state, results)

            states = states.write(i, next_state)
            accepts = accepts.write(i, next_results.inner_results.inner_results.is_accepted)
            loglkl = loglkl.write(i, next_results.inner_results.inner_results.accepted_results.target_log_prob)

            return i + 1, next_state, next_results, states, accepts, loglkl

        _, state, results, states, accepts, loglkl = tf.while_loop(
            lambda i, *_: i < num_steps,
            loop_vars=[0, current_state, kernel_results, states, accepts, loglkl],
            body=body,
            parallel_iterations=1,
        )

        return state, results, states.stack(), accepts.stack(), loglkl.stack()

    if inner_level == 1:
        run_chunk = run_chunk1
    elif inner_level == 2:
        run_chunk = run_chunk2
    else:
        raise NotImplementedError("Using more than 2 nested kernels is not supported.")

    total_steps = n_steps + num_burnin_steps
    chunk_size = min(max(1, total_steps // 100), 1000)
    chunk_remainder = total_steps % chunk_size

    # pre-compilation
    _, res, _, _, _ = run_chunk(
        current_state, kernel_results, chunk_size
    )
    _, res, _, _, _ = run_chunk(
        current_state, res, chunk_size
    )
    if chunk_remainder > 0:
        _, res, _, _, _ = run_chunk(
            current_state, res, chunk_remainder
        )

    samples_list = []
    accepts_list = []
    loglkl_list = []

    steps_done = 0

    if progress_bar:
        py_update(
            0,
            num_samples=n_steps,
            num_burnin_steps=num_burnin_steps,
            num_steps_between_results=0
        )

    while steps_done < total_steps:
        steps_this = min(chunk_size, total_steps - steps_done)
        current_state, kernel_results, chunk_states, accepts, loglkl = run_chunk(
            current_state,
            kernel_results,
            steps_this
        )

        samples_list.append(chunk_states)
        accepts_list.append(accepts)
        loglkl_list.append(loglkl)

        steps_done += steps_this

        if progress_bar:
            py_update(
                steps_done,
                num_samples=n_steps,
                num_burnin_steps=num_burnin_steps,
                num_steps_between_results=0
            )

    samples = tf.concat(samples_list, axis=0)
    samples = samples[num_burnin_steps:]

    accepts = tf.concat(accepts_list, axis=0)
    accepts = accepts[num_burnin_steps:]

    loglkl = tf.concat(loglkl_list, axis=0)
    loglkl = loglkl[num_burnin_steps:]

    acceptance_rate = tf.reduce_mean(tf.cast(accepts, tf.float32))
    return samples, loglkl, acceptance_rate


def mh_proposal_fn(state, seed, step_size=0.1):
    flat_state = tf.nest.flatten(state)
    n = len(flat_state)

    seeds = tf.random.experimental.stateless_split(seed, n)

    flat_next = []
    for i in range(n):

        s = flat_state[i]
        seed_i = seeds[i]

        flat_next.append(
            s + tf.random.stateless_normal(
                tf.shape(s),
                seed=seed_i,
                stddev=step_size
            )
        )

    return tf.nest.pack_sequence_as(state, flat_next)


MalaResults = namedtuple(
    "MalaResults",
    ["inner_results", "step_size"]
)

class MalaWithStepSize(tfp.mcmc.TransitionKernel):
    def __init__(self, target_log_prob_fn, step_size, volatility_fn):
        self._target_log_prob_fn = target_log_prob_fn
        self._volatility_fn = volatility_fn
        self._step_size = step_size

    @property
    def is_calibrated(self):
        return True

    def one_step(self, current_state, previous_kernel_results, seed=None):
        step_size = previous_kernel_results.step_size

        kernel = tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
            target_log_prob_fn=self._target_log_prob_fn,
            step_size=step_size,
            volatility_fn=self._volatility_fn,
        )

        new_state, inner_results = kernel.one_step(
            current_state,
            previous_kernel_results.inner_results,
            seed=seed
        )

        return new_state, MalaResults(
            inner_results=inner_results,
            step_size=step_size
        )

    def bootstrap_results(self, init_state):
        kernel = tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
            target_log_prob_fn=self._target_log_prob_fn,
            step_size=self._step_size,
            volatility_fn=self._volatility_fn,
        )

        inner_results = kernel.bootstrap_results(init_state)

        return MalaResults(
            inner_results=inner_results,
            step_size=tf.convert_to_tensor(self._step_size, tf.float32)
        )

def py_update(step, num_samples, num_burnin_steps, num_steps_between_results):
    global start_time
    if step == 0:
        start_time = time.time()
        return 0.0
    else:
        now_time = time.time()
        elapsed = now_time - start_time
        step = step // (num_steps_between_results + 1) + 1*int(num_steps_between_results > 0)

    step = int(step)
    start_time = float(start_time)
    burnin = float(num_burnin_steps)
    total = float(num_samples + num_burnin_steps)
    progress = step / total


    # --- PERCENT ---
    percent_value = int(progress * 100)
    percent = f"{percent_value}%"
    if len(percent) < 3:
        percent = "  " + percent
    elif len(percent) < 4:
        percent = " " + percent


    # --- COUNTER ---
    len_total = len(str(int(total)))
    len_step = len(str(step))
    diff_counter = len_total - len_step
    counter = " "*diff_counter + f"{step}/{int(total)}"

    # --- RATE ---
    rate = max((step) / max(elapsed, 1e-10), 1e-3)
    eta = (total - step) / rate

    # format time as mm:ss
    def format_time(seconds_total):
        minutes = int(seconds_total // 60)
        seconds = int(seconds_total % 60)

        minutes_str = str(minutes)
        seconds_str = str(seconds)

        # zero-pad manually
        if len(minutes_str) < 2:
            minutes_str = "0" + minutes_str
        if len(seconds_str) < 2:
            seconds_str = "0" + seconds_str

        return f"{minutes_str}:{seconds_str}"

    elapsed_str = format_time(elapsed)
    eta_str = format_time(eta)

    # --- RATE STRING ---
    rate_value = round(rate * 100) / 100
    rate_str = f"{rate_value:.2f}"
    len_rate = len(rate_str)
    len_rate_max = 6
    diff_rate = len_rate_max - len_rate
    rate_str = " "*diff_rate + rate_str + " it/s"

    # --- BAR ---
    try:
        output_size = os.get_terminal_size().columns
        bar_width = max(output_size - len("".join([percent,counter,elapsed_str,eta_str,rate_str])) - 9, 10)
    except:
        bar_width = 10
    filled = round(progress * bar_width)
    filled_burnin = round(min(burnin / total, progress) * bar_width)
    filled_sampling = filled - filled_burnin

    empty = bar_width - filled

    bar = "\033[93m█\033[0m" * filled_burnin + "█" * filled_sampling + " " * empty

    line = "".join([
            percent, "|",
            bar, "| ",
            counter,
            " [",
            elapsed_str, "<", eta_str, ", ",
            rate_str,
            "]"
    ])

    sys.stdout.write("\r" + line)
    sys.stdout.flush()
    if step == total:
        sys.stdout.write("\n")

    return 0.0
