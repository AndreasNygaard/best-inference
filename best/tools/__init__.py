import time
import os
import sys
import shutil
from collections import namedtuple
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

try:
    output_size = os.get_terminal_size().columns
except:
    output_size = 0

class LogProbCounter:
    def __init__(self, log_prob_fn):
        self.log_prob_fn = log_prob_fn
        self.num_calls = tf.Variable(0, dtype=tf.int64, trainable=False)

    def __call__(self, *args):
        self.num_calls.assign_add(1)
        return self.log_prob_fn(*args)


def jit_tfp_sample(n_steps, num_burnin_steps, current_state, kernel, progress_bar=True, jit_compile=True):
    kernel_results = kernel.bootstrap_results(current_state)

    def get_target_log_prob(results):
        if hasattr(results, "accepted_results"):
            return results.accepted_results.target_log_prob
        elif hasattr(results, "target_log_prob"):
            return results.target_log_prob
        elif hasattr(results, "inner_results"):
            return get_target_log_prob(results.inner_results)
        else:
            raise ValueError(f"Cannot find target_log_prob in {type(results)}")

    def get_acceptance(results):
        if hasattr(results, "is_accepted"):
            return results.is_accepted

        if hasattr(results, "log_accept_ratio"):
            return tf.exp(tf.minimum(0., results.log_accept_ratio)) > 0.5

        if hasattr(results, "inner_results"):
            return get_acceptance(results.inner_results)

        raise ValueError(f"Cannot find acceptance in {type(results)}")

    @tf.function(jit_compile=jit_compile)
    def run_chunk(current_state, kernel_results, num_steps):
        states = tf.TensorArray(current_state.dtype, size=num_steps)
        accepts = tf.TensorArray(tf.bool, size=num_steps)
        loglkl = tf.TensorArray(current_state.dtype, size=num_steps)

        def body(i, state, results, states, accepts, loglkl):
            next_state, next_results = kernel.one_step(state, results)

            states = states.write(i, next_state)
            accepts = accepts.write(i, get_acceptance(next_results))
            loglkl = loglkl.write(i, get_target_log_prob(next_results))

            return i + 1, next_state, next_results, states, accepts, loglkl

        _, state, results, states, accepts, loglkl = tf.while_loop(
            lambda i, *_: i < num_steps,
            loop_vars=[0, current_state, kernel_results, states, accepts, loglkl],
            body=body,
            parallel_iterations=1,
        )

        return state, results, states.stack(), accepts.stack(), loglkl.stack()

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
    if output_size > 0:
        bar_width = max(output_size - len("".join([percent,counter,elapsed_str,eta_str,rate_str])) - 9, 10)
    else:
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




class AdaptiveClusterTree:
    def __init__(self, efficiency=0.8, max_depth=5, min_cluster_size=50, merge_tolerance=0.10):
        self.efficiency = tf.convert_to_tensor(efficiency, dtype=tf.float32)
        self.max_depth = tf.convert_to_tensor(max_depth, dtype=tf.int32)
        self.min_cluster_size = tf.convert_to_tensor(min_cluster_size, dtype=tf.int32)
        self.merge_tolerance = tf.convert_to_tensor(merge_tolerance, dtype=tf.float32)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def initialise_tree(self, points):
        self.points = points
        N = tf.shape(points)[0]
        dim = tf.shape(points)[1]
        root_indices = tf.ones_like(points[:, 0], dtype=tf.bool)
        root_volume, root_center, root_cov, root_inv, root_chol = self.get_ellipsoid(points, root_indices)
        nodes_indices = tf.zeros((2**(self.max_depth+1)-1, N), dtype=tf.bool)
        nodes_indices = tf.tensor_scatter_nd_update(nodes_indices, [[0]], [root_indices])
        nodes_volumes = tf.zeros((2**(self.max_depth+1)-1,), dtype=tf.float32)
        nodes_volumes = tf.tensor_scatter_nd_update(nodes_volumes, [[0]], [root_volume])
        nodes_centers = tf.zeros((2**(self.max_depth+1)-1, dim), dtype=tf.float32)
        nodes_centers = tf.tensor_scatter_nd_update(nodes_centers, [[0]], [root_center])
        nodes_covs = tf.zeros((2**(self.max_depth+1)-1, dim, dim), dtype=tf.float32)
        nodes_covs = tf.tensor_scatter_nd_update(nodes_covs, [[0]], [root_cov])
        nodes_invs = tf.zeros((2**(self.max_depth+1)-1, dim, dim), dtype=tf.float32)
        nodes_invs = tf.tensor_scatter_nd_update(nodes_invs, [[0]], [root_inv])
        nodes_chols = tf.zeros((2**(self.max_depth+1)-1, dim, dim), dtype=tf.float32)
        nodes_chols = tf.tensor_scatter_nd_update(nodes_chols, [[0]], [root_chol])
        nodes_accept = tf.ones((2**(self.max_depth+1)-1,), dtype=tf.bool)
        return nodes_indices, nodes_volumes, nodes_centers, nodes_covs, nodes_invs, nodes_chols, nodes_accept

    @tf.function(jit_compile=True, reduce_retracing=True)
    def build_cluster_tree(self, points, nodes_indices, nodes_volumes, nodes_centers, nodes_covs, nodes_invs, nodes_chols):

        def depth_cond(depth, running_index, nodes_indices, nodes_volumes, nodes_centers, nodes_covs, nodes_invs, nodes_chols):
            return depth < self.max_depth

        def depth_body(depth, running_index, nodes_indices, nodes_volumes, nodes_centers, nodes_covs, nodes_invs, nodes_chols):

            n_nodes = tf.pow(2, depth)
            i_start = running_index
            i_end = running_index + n_nodes

            def node_cond(i, running_index, nodes_indices, nodes_volumes, nodes_centers, nodes_covs, nodes_invs, nodes_chols):
                return i < i_end

            def node_body(i, running_index, nodes_indices, nodes_volumes, nodes_centers, nodes_covs, nodes_invs, nodes_chols):

                node_indices = nodes_indices[i]

                mask, _, _ = self.binary_kmeans(points, node_indices)
                #mask = tf.concat([tf.ones((400,), dtype=tf.int32), tf.zeros((500,), dtype=tf.int32)], axis=0)
                indices1 = tf.logical_and(node_indices, mask == 1)
                indices2 = tf.logical_and(node_indices, mask == 0)

                child1_idx = 2 * i + 1
                child2_idx = 2 * i + 2

                # Update child indices
                nodes_indices = tf.tensor_scatter_nd_update(
                    nodes_indices,
                    tf.reshape(
                        tf.stack([child1_idx, child2_idx]), 
                        (-1, 1)
                    ),
                    tf.stack([indices1, indices2])
                )

                N1 = tf.reduce_sum(tf.cast(indices1, tf.int32))
                N2 = tf.reduce_sum(tf.cast(indices2, tf.int32))
                def compute_ellipsoid(indices):
                    return self.get_ellipsoid(points, indices)

                volume1, center1, cov1, inv1, chol1 = tf.cond(
                    N1 > 0,
                    lambda: compute_ellipsoid(indices1),
                    lambda: (tf.constant(0.0, dtype=points.dtype),
                             tf.zeros((points.shape[1],), dtype=points.dtype),
                             tf.zeros((points.shape[1], points.shape[1]), dtype=points.dtype),
                             tf.zeros((points.shape[1], points.shape[1]), dtype=points.dtype),
                             tf.zeros((points.shape[1], points.shape[1]), dtype=points.dtype)
                    )
                )

                volume2, center2, cov2, inv2, chol2 = tf.cond(
                    N2 > 0,
                    lambda: compute_ellipsoid(indices2),
                    lambda: (tf.constant(0.0, dtype=points.dtype),
                             tf.zeros((points.shape[1],), dtype=points.dtype),
                             tf.zeros((points.shape[1], points.shape[1]), dtype=points.dtype),
                             tf.zeros((points.shape[1], points.shape[1]), dtype=points.dtype),
                             tf.zeros((points.shape[1], points.shape[1]), dtype=points.dtype)
                    )
                )

                nodes_volumes = tf.tensor_scatter_nd_update(
                    nodes_volumes,
                    tf.reshape(
                        tf.stack([child1_idx, child2_idx]),
                        (-1, 1)
                    ),
                    tf.stack([volume1, volume2])
                )

                nodes_centers = tf.tensor_scatter_nd_update(
                    nodes_centers,
                    tf.reshape(
                        tf.stack([child1_idx, child2_idx]),
                        (-1, 1)
                    ),
                    tf.stack([center1, center2])
                )

                nodes_covs = tf.tensor_scatter_nd_update(
                    nodes_covs,
                    tf.reshape(
                        tf.stack([child1_idx, child2_idx]),
                        (-1, 1)
                    ),
                    tf.stack([cov1, cov2])
                )

                nodes_invs = tf.tensor_scatter_nd_update(
                    nodes_invs,
                    tf.reshape(
                        tf.stack([child1_idx, child2_idx]),
                        (-1, 1)
                    ),
                    tf.stack([inv1, inv2])
                )

                nodes_chols = tf.tensor_scatter_nd_update(
                    nodes_chols,
                    tf.reshape(
                        tf.stack([child1_idx, child2_idx]),
                        (-1, 1)
                    ),
                    tf.stack([chol1, chol2])
                )

                return i + 1, running_index + 1, nodes_indices, nodes_volumes, nodes_centers, nodes_covs, nodes_invs, nodes_chols

            _, running_index, nodes_indices, nodes_volumes, nodes_centers, nodes_covs, nodes_invs, nodes_chols = tf.while_loop(
                node_cond,
                node_body,
                (i_start, running_index, nodes_indices, nodes_volumes, nodes_centers, nodes_covs, nodes_invs, nodes_chols)
            )

            return depth + 1, running_index, nodes_indices, nodes_volumes, nodes_centers, nodes_covs, nodes_invs, nodes_chols


        _, _, nodes_indices, nodes_volumes, nodes_centers, nodes_covs, nodes_invs, nodes_chols = tf.while_loop(
            depth_cond,
            depth_body,
            (tf.constant(0), tf.constant(0), nodes_indices, nodes_volumes, nodes_centers, nodes_covs, nodes_invs, nodes_chols)
        )

        return nodes_indices, nodes_volumes, nodes_centers, nodes_covs, nodes_invs, nodes_chols

    @tf.function(jit_compile=True, reduce_retracing=True)
    def prune_tree(self, nodes_indices, nodes_volumes, nodes_centers, nodes_covs, nodes_invs, nodes_chols, nodes_accept):

        def cond(i, nodes_indices, nodes_volumes, nodes_centers, nodes_covs, nodes_invs, nodes_chols, nodes_accept):
            return i >= 0

        def body(i, nodes_indices, nodes_volumes, nodes_centers, nodes_covs, nodes_invs, nodes_chols, nodes_accept):

            child1_idx = 2 * i + 1
            child2_idx = 2 * i + 2

            child_volume = (
                nodes_volumes[child1_idx] +
                nodes_volumes[child2_idx]
            )

            volume = nodes_volumes[i]
            
            N1 = tf.reduce_sum(tf.cast(nodes_indices[child1_idx], tf.int32))
            N2 = tf.reduce_sum(tf.cast(nodes_indices[child2_idx], tf.int32))
            not_shrink = volume < (1 + self.merge_tolerance) * child_volume
            too_small_1 = N1 < self.min_cluster_size
            too_small_2 = N2 < self.min_cluster_size
            too_small = tf.logical_or(too_small_1, too_small_2)
            child_accept_1 = nodes_accept[child1_idx]
            child_accept_2 = nodes_accept[child2_idx]
            both_children_accept = tf.logical_and(child_accept_1, child_accept_2)
            merge = tf.logical_and(tf.logical_or(not_shrink, too_small), both_children_accept)

            # Update parent acceptance
            nodes_accept = tf.tensor_scatter_nd_update(
                nodes_accept,
                tf.reshape(i, (1, 1)),
                tf.reshape(merge, (1,))
            )

            # Update children only if merged
            def update_children(nodes_accept):
                nodes_accept = tf.tensor_scatter_nd_update(
                    nodes_accept,
                    tf.reshape(
                        tf.stack([child1_idx, child2_idx]),
                        (-1, 1)
                    ),
                    tf.constant([False, False])
                )
                return nodes_accept

            nodes_accept = tf.cond(
                merge,
                lambda: update_children(nodes_accept),
                lambda: nodes_accept
            )

            N_accepted = tf.reduce_sum(
                tf.cast(nodes_accept, tf.int32)
            )

            def update_root_false(nodes_accept):
                nodes_accept = tf.tensor_scatter_nd_update(
                    nodes_accept,
                    tf.reshape(0, (1, 1)),
                    tf.reshape(False, (1,))
                )
                return nodes_accept

            def update_root_true(nodes_accept):
                nodes_accept = tf.tensor_scatter_nd_update(
                    nodes_accept,
                    tf.reshape(0, (1, 1)),
                    tf.reshape(True, (1,))
                )
                return nodes_accept
            
            nodes_accept = tf.cond(
                tf.logical_and(N_accepted > 1, i == 0),
                lambda: update_root_false(nodes_accept),
                lambda: update_root_true(nodes_accept)
            )

            return i - 1, nodes_indices, nodes_volumes, nodes_centers, nodes_covs, nodes_invs, nodes_chols, nodes_accept

        _, nodes_indices, nodes_volumes, nodes_centers, nodes_covs, nodes_invs, nodes_chols, nodes_accept = tf.while_loop(
            cond,
            body,
            (2**self.max_depth - 2, nodes_indices, nodes_volumes, nodes_centers, nodes_covs, nodes_invs, nodes_chols, nodes_accept)
        )

        return nodes_indices, nodes_volumes, nodes_centers, nodes_covs, nodes_invs, nodes_chols, nodes_accept

    @tf.function(jit_compile=True, reduce_retracing=True)
    def get_ellipsoid(self, points, mask):

        dtype = points.dtype

        # Convert mask to weights
        weights = tf.cast(mask, dtype)

        # Number of active points
        n_points = tf.reduce_sum(weights)

        # Prevent division by zero
        n_points = tf.maximum(
            n_points,
            tf.constant(1, dtype)
        )

        center = (
            tf.reduce_sum(
                points * weights[:, None],
                axis=0
            )
            /
            n_points
        )

        delta = points - center

        # Remove inactive points
        delta_masked = delta * weights[:, None]

        cov = tf.matmul(
            delta_masked,
            delta_masked,
            transpose_a=True
        ) / n_points


        cov += (
            tf.eye(
                tf.shape(cov)[0],
                dtype=dtype
            )
            *
            tf.constant(
                1e-12,
                dtype
            )
        )


        chol = tf.linalg.cholesky(cov)

        y = tf.linalg.triangular_solve(
            chol,
            tf.transpose(delta_masked)
        )

        maha = tf.reduce_sum(
            y * y,
            axis=0
        )

        # Ignore inactive points
        maha = tf.where(
            mask,
            maha,
            tf.constant(
                -tf.float32.max,
                dtype=dtype
            )
        )

        radius2 = tf.reduce_max(maha)


        enlargement = self.adaptive_enlargement(
            tf.cast(tf.reduce_sum(tf.cast(mask, tf.int32)), tf.int32),
            self.efficiency
        )


        cov = (
            cov
            *
            radius2
            *
            tf.cast(enlargement, dtype)
        )

        chol = tf.linalg.cholesky(cov)

        inv=tf.linalg.inv(cov)

        ndim = tf.cast(
            tf.shape(cov)[0],
            dtype
        )

        log_det_half = tf.reduce_sum(
            tf.math.log(
                tf.linalg.diag_part(chol)
            )
        )

        unit_ball = (
            tf.pow(
                tf.constant(np.pi, dtype=dtype),
                ndim / 2
            )
            /
            tf.exp(
                tf.math.lgamma(
                    ndim / 2 + 1
                )
            )
        )
        
        volume = unit_ball * tf.exp(log_det_half)


        return volume, center, cov, inv, chol


    @tf.function(jit_compile=True, reduce_retracing=True)
    def adaptive_enlargement(
        self,
        npoints,
        efficiency
    ):
        """
        MultiNest-inspired enlargement.

        npoints:
            number of points in cluster

        efficiency:
            target sampling efficiency
        """

        npoints = tf.cast(
            npoints,
            tf.float32
        )

        factor = (
            1.0
            +
            tf.sqrt(
                2.0/npoints
            )
        )**2

        factor /= efficiency

        return factor
    
    @tf.function(jit_compile=True, reduce_retracing=True)
    def binary_kmeans(
        self,
        points,
        active_mask,
        max_iter=20,
    ):
        """
        Binary k-means on a masked subset of points.

        Parameters
        ----------
        points : (N, ndim)
            Full point set.

        active_mask : (N,)
            Boolean mask indicating which points participate.

        Returns
        -------
        labels : (N,)
            Labels for all points. Labels outside active_mask are 0.

        centres : (2, ndim)

        sizes : (2,)
            Number of active points in each cluster.
        """

        dtype = points.dtype

        weights = tf.cast(active_mask, dtype)

        n_active = tf.maximum(
            tf.reduce_sum(weights),
            tf.constant(1., dtype)
        )

        # --------------------------------------------------------
        # Initial centre
        # --------------------------------------------------------

        centre0 = (
            tf.reduce_sum(
                points * weights[:, None],
                axis=0
            ) / n_active
        )

        diff = points - centre0

        dist2 = tf.reduce_sum(
            diff * diff,
            axis=1
        )

        # Ignore inactive points when choosing the furthest point
        dist2 = tf.where(
            active_mask,
            dist2,
            tf.fill(tf.shape(dist2), tf.constant(-1e30, dtype))
        )

        index1 = tf.argmax(
            dist2,
            output_type=tf.int32
        )

        centre1 = tf.gather(points, index1)

        centres = tf.stack(
            [centre0, centre1],
            axis=0
        )

        # --------------------------------------------------------
        # Lloyd iterations
        # --------------------------------------------------------

        def cond(i, centres):
            return i < max_iter

        def body(i, centres):

            delta = (
                tf.expand_dims(points, 1)
                - tf.expand_dims(centres, 0)
            )

            d2 = tf.reduce_sum(
                delta * delta,
                axis=2
            )

            labels = tf.argmin(
                d2,
                axis=1,
                output_type=tf.int32
            )

            # Ignore inactive points
            labels = tf.where(
                active_mask,
                labels,
                tf.zeros_like(labels)
            )

            mask0 = tf.cast(
                tf.logical_and(active_mask, labels == 0),
                dtype
            )

            mask1 = tf.cast(
                tf.logical_and(active_mask, labels == 1),
                dtype
            )

            n0 = tf.maximum(
                tf.reduce_sum(mask0),
                tf.constant(1., dtype)
            )

            n1 = tf.maximum(
                tf.reduce_sum(mask1),
                tf.constant(1., dtype)
            )

            centre0 = tf.cond(
                tf.reduce_sum(mask0) > 0,
                lambda: tf.reduce_sum(
                    points * mask0[:, None],
                    axis=0
                ) / n0,
                lambda: centres[0]
            )

            centre1 = tf.cond(
                tf.reduce_sum(mask1) > 0,
                lambda: tf.reduce_sum(
                    points * mask1[:, None],
                    axis=0
                ) / n1,
                lambda: centres[1]
            )

            centres = tf.stack(
                [centre0, centre1]
            )

            return i + 1, centres

        _, centres = tf.while_loop(
            cond,
            body,
            (
                tf.constant(0),
                centres
            )
        )

        # --------------------------------------------------------
        # Final assignment
        # --------------------------------------------------------

        delta = (
            tf.expand_dims(points, 1)
            - tf.expand_dims(centres, 0)
        )

        d2 = tf.reduce_sum(
            delta * delta,
            axis=2
        )

        labels = tf.argmin(
            d2,
            axis=1,
            output_type=tf.int32
        )

        labels = tf.where(
            active_mask,
            labels,
            tf.zeros_like(labels)
        )

        sizes = tf.stack([
            tf.reduce_sum(
                tf.cast(
                    tf.logical_and(active_mask, labels == 0),
                    tf.int32
                )
            ),
            tf.reduce_sum(
                tf.cast(
                    tf.logical_and(active_mask, labels == 1),
                    tf.int32
                )
            )
        ])

        return labels, centres, sizes


    @tf.function(jit_compile=True, reduce_retracing=True)
    def get_all_clusters(self, points):
        nodes_indices, nodes_volumes, nodes_centers, nodes_covs, nodes_invs, nodes_chols, nodes_accept = self.initialise_tree(points)
        nodes_indices, nodes_volumes, nodes_centers, nodes_covs, nodes_invs, nodes_chols = self.build_cluster_tree(points, nodes_indices, nodes_volumes, nodes_centers, nodes_covs, nodes_invs, nodes_chols)
        nodes_indices, nodes_volumes, nodes_centers, nodes_covs, nodes_invs, nodes_chols, nodes_accept = self.prune_tree(nodes_indices, nodes_volumes, nodes_centers, nodes_covs, nodes_invs, nodes_chols, nodes_accept)
        return nodes_accept, nodes_indices, nodes_volumes, nodes_centers, nodes_covs, nodes_invs, nodes_chols

    def get_accepted_clusters(self, points):
        t0 = time()
        nodes_accept, nodes_indices, nodes_volumes, nodes_centers, nodes_covs, nodes_invs, nodes_chols = self.get_all_clusters(points)
        t1 = time()
        clusters = [tf.gather(points, tf.where(nodes_indices[i])[:, 0]) for i in range(nodes_accept.shape[0]) if nodes_accept[i]]
        volumes = [nodes_volumes[i] for i in range(nodes_accept.shape[0]) if nodes_accept[i]]
        centers = [nodes_centers[i] for i in range(nodes_accept.shape[0]) if nodes_accept[i]]
        covs = [nodes_covs[i] for i in range(nodes_accept.shape[0]) if nodes_accept[i]]
        invs = [nodes_invs[i] for i in range(nodes_accept.shape[0]) if nodes_accept[i]]
        chols = [nodes_chols[i] for i in range(nodes_accept.shape[0]) if nodes_accept[i]]
        t2 = time()
        print(f"Adaptive clustering took {t1 - t0:.4f} seconds.")
        print(f"Extracting accepted clusters took {t2 - t1:.4f} seconds.")
        return clusters, volumes, centers, covs, invs, chols




class slice_sampler:
    def __init__(self, loglike, bounds, n_iter=10, max_expand=20, max_shrink=100, buffer_size=100, expand_to_worst=False):
        self.loglike = loglike
        self.lower, self.upper = bounds
        self.n_iter = n_iter
        self.max_expand = max_expand
        self.max_shrink = max_shrink
        self.buffer_size = buffer_size
        self.expand_to_worst = expand_to_worst
        self.expand_to_worst_tf = tf.convert_to_tensor(expand_to_worst, dtype=tf.bool)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def sample(
        self,
        x0s,
        chols,
        worst_logLs,
        seed=tf.constant([123,456], tf.int32)
    ):

        dtype = x0s.dtype
        n_updates = x0s.shape[0]
        ndim      = x0s.shape[1]
        buffer_points = tf.zeros((self.buffer_size, n_updates, ndim), dtype=dtype)
        buffer_logL   = tf.ones((self.buffer_size, n_updates), dtype=dtype)*(-dtype.max)

        expand_compare_logLs = tf.cond(
            self.expand_to_worst_tf,
            lambda: tf.ones_like(worst_logLs) * worst_logLs[0],
            lambda: worst_logLs
        )

        # =====================================================
        # outer slice iterations
        # =====================================================

        def outer_cond(i, x0s, buffer_points, buffer_logL):
            return i < self.n_iter

        def outer_body(i, x0s, buffer_points, buffer_logL):

            # -------------------------------------------------
            # random direction
            # -------------------------------------------------

            seed2 = seed + tf.stack([i+1, tf.range(n_updates)[0]+1])

            dir_normal = tf.random.stateless_normal(
                (n_updates, ndim),
                dtype=dtype,
                seed=seed2
            )

            dir_normal /= tf.linalg.norm(
                dir_normal,
                axis=1,
                keepdims=True,
            )

            vs = tf.einsum(
                "bij,bj->bi",
                chols,
                dir_normal,
            )

            def x(t):
                return x0s + t * vs
            
            def inside_prior(x):
                return tf.reduce_all(
                    tf.logical_and(
                        x >= tf.cast(self.lower, dtype),
                        x <= tf.cast(self.upper, dtype),
                    ),
                    axis=1,
                )

            # -------------------------------------------------
            # initial bracket
            # -------------------------------------------------

            a = tf.ones(
                (n_updates, 1),
                dtype=dtype,
            ) * -2.0*tf.sqrt(tf.cast(ndim, dtype))

            b = tf.ones(
                (n_updates, 1),
                dtype=dtype,
            ) * 2.0*tf.sqrt(tf.cast(ndim, dtype))

            # -------------------------------------------------
            # step-out
            # -------------------------------------------------

            x_a = x(a)
            x_b = x(b)

            inside_a = inside_prior(x_a)
            inside_b = inside_prior(x_b)

            logL = self.loglike(tf.concat([x_a, x_b], axis=0))
            logL_a = logL[:n_updates]
            logL_b = logL[n_updates:]

            grow_a = tf.cast(
                tf.logical_not(
                    tf.logical_or(
                        logL_a < expand_compare_logLs,
                        tf.logical_not(inside_a),
                    )
                ),
                dtype,
            )[:, None]

            grow_b = tf.cast(
                tf.logical_not(
                    tf.logical_or(
                        logL_b < expand_compare_logLs,
                        tf.logical_not(inside_b),
                    )
                ),
                dtype,
            )[:, None]

            def expand_cond(
                k,
                a,
                b,
                grow_a,
                grow_b,
            ):

                return tf.logical_and(
                    k < self.max_expand,
                    tf.logical_or(
                        tf.reduce_any(grow_a > 0),
                        tf.reduce_any(grow_b > 0),
                    ),
                )

            def expand_body(
                k,
                a,
                b,
                grow_a,
                grow_b,
            ):

                a_new = a + a * grow_a
                b_new = b + b * grow_b

                x_a = x(a_new)
                x_b = x(b_new)

                inside_a = inside_prior(x_a)
                inside_b = inside_prior(x_b)

                logL = self.loglike(tf.concat([x_a, x_b], axis=0))
                logL_a = logL[:n_updates]
                logL_b = logL[n_updates:]

                grow_a = tf.cast(
                    tf.logical_not(
                        tf.logical_and(
                            logL_a < expand_compare_logLs,
                            tf.logical_not(inside_a),
                        )
                    ),
                    dtype,
                )[:, None]

                grow_b = tf.cast(
                    tf.logical_not(
                        tf.logical_and(
                            logL_b < expand_compare_logLs,
                            tf.logical_not(inside_b),
                        )
                    ),
                    dtype,
                )[:, None]

                # If the expansion went outside the prior,
                # stop expansion in that direction.
                a = (
                    tf.cast(inside_a[:,None], dtype)
                    * a_new
                    +
                    tf.cast(
                        tf.logical_not(inside_a)[:,None],
                        dtype,
                    )
                    * a
                )

                b = (
                    tf.cast(inside_b[:,None], dtype)
                    * b_new
                    +
                    tf.cast(
                        tf.logical_not(inside_b)[:,None],
                        dtype,
                    )
                    * b
                )

                return (
                    k + 1,
                    a,
                    b,
                    grow_a,
                    grow_b,
                )

            _, a, b, _, _ = tf.while_loop(
                expand_cond,
                expand_body,
                (
                    tf.constant(0),
                    a,
                    b,
                    grow_a,
                    grow_b,
                ),
            )

            # -------------------------------------------------
            # shrinkage
            # -------------------------------------------------

            accepted = tf.zeros(
                (n_updates, 1),
                dtype=dtype,
            )

            t_final = tf.zeros(
                (n_updates, 1),
                dtype=dtype,
            )

            def shrink_cond(
                k,
                a,
                b,
                accepted,
                t_final,
                buffer_points,
                buffer_logL,
            ):

                return tf.logical_and(
                    k < self.max_shrink,
                    tf.reduce_sum(accepted)
                    <
                    tf.cast(n_updates, dtype),
                )

            def shrink_body(
                k,
                a,
                b,
                accepted,
                t_final,
                buffer_points,
                buffer_logL,
            ):

                seed3 = seed + tf.stack([i+k+1, tf.range(n_updates)[0]+1])

                t = tf.random.stateless_uniform(
                    (n_updates, 1),
                    minval=a,
                    maxval=b,
                    dtype=dtype,
                    seed=seed3
                )

                x_t = x(t)

                inside = inside_prior(x_t)

                logL_t = self.loglike(x_t)

                good = tf.cast(
                    tf.logical_and(
                        logL_t > worst_logLs,
                        inside,
                    ),
                    dtype,
                )[:, None]

                new_accept = (
                    (1.0 - accepted)
                    *
                    good
                )

                t_final = (
                    (1.0 - new_accept)
                    * t_final
                    +
                    new_accept
                    * t
                )

                accepted = tf.maximum(
                    accepted,
                    good,
                )

                # -----------------------------------------
                # only update intervals for chains
                # not yet accepted
                # -----------------------------------------

                active = 1.0 - accepted

                t_neg = tf.cast(
                    t < 0.0,
                    dtype,
                )

                t_pos = 1.0 - t_neg

                bad = 1.0 - good

                update_a = (
                    active
                    * bad
                    * t_neg
                )

                update_b = (
                    active
                    * bad
                    * t_pos
                )

                a = (
                    (1.0 - update_a)
                    * a
                    +
                    update_a
                    * t
                )

                b = (
                    (1.0 - update_b)
                    * b
                    +
                    update_b
                    * t
                )

                def save_shrink_history():
                    # update buffer on index k mod buffer_size
                    update_index = tf.math.floormod(k, self.buffer_size)
                    new_points = x_t
                    new_logL = logL_t
                    buffer_points_new = tf.tensor_scatter_nd_update(
                        buffer_points,
                        tf.reshape(update_index, (1, 1)),
                        tf.reshape(new_points, (1, n_updates, ndim))
                    )
                    buffer_logL_new = tf.tensor_scatter_nd_update(
                        buffer_logL,
                        tf.reshape(update_index, (1, 1)),
                        tf.reshape(new_logL, (1, n_updates))
                    )
                    return buffer_points_new, buffer_logL_new


                buffer_points, buffer_logL = tf.cond(
                    tf.equal(i, self.n_iter-1),
                    save_shrink_history,
                    lambda: (buffer_points, buffer_logL)
                )

                return (
                    k + 1,
                    a,
                    b,
                    accepted,
                    t_final,
                    buffer_points,
                    buffer_logL
                )

            _, _, _, _, t_final, buffer_points, buffer_logL = tf.while_loop(
                shrink_cond,
                shrink_body,
                (
                    tf.constant(0),
                    a,
                    b,
                    accepted,
                    t_final,
                    buffer_points,
                    buffer_logL
                ),
            )

            x0s = x(t_final)

            return (
                i + 1,
                x0s,
                buffer_points,
                buffer_logL
            )

        _, x0s, buffer_points, buffer_logL = tf.while_loop(
            outer_cond,
            outer_body,
            (
                tf.constant(0),
                x0s,
                buffer_points,
                buffer_logL
            ),
        )

        return x0s, buffer_points, buffer_logL




@tf.function(jit_compile=True, reduce_retracing=True)
def safe_cholesky(covs, nodes_accept, dim, max_iter=20):

    dtype = covs.dtype
    n_cluster = covs.shape[0]
    ndim = dim

    eye = tf.linalg.diag(tf.ones((ndim,), dtype=dtype))

    # -----------------------------------------------------
    # Symmetrize
    # -----------------------------------------------------

    covs = 0.5 * (
        covs
        +
        tf.transpose(
            covs,
            perm=[0,2,1]
        )
    )


    # -----------------------------------------------------
    # Remove inactive clusters completely
    # -----------------------------------------------------

    active = tf.cast(
        nodes_accept[:, None, None],
        dtype,
    )

    covs = (
        covs * active
        +
        (1.0 - active) * eye[None,:,:]
    )


    # -----------------------------------------------------
    # First attempt
    # -----------------------------------------------------

    chol = tf.linalg.cholesky(covs)

    bad = tf.reduce_any(
        tf.math.is_nan(chol),
        axis=[1,2],
    )


    # only active clusters can be bad
    bad = tf.logical_and(
        bad,
        nodes_accept,
    )


    # -----------------------------------------------------
    # Iterative repair
    # -----------------------------------------------------

    def cond(i, covs, chol, bad):
        return tf.logical_and(
            i < max_iter,
            tf.reduce_any(bad),
        )


    def body(i, covs, chol, bad):

        # eigendecomposition of all matrices
        # (inactive ones are identity so harmless)
        eigvals, eigvecs = tf.linalg.eigh(covs)


        eigvals = tf.maximum(
            eigvals,
            tf.cast(0.0, dtype)
        )


        covs = (
            eigvecs
            @
            tf.linalg.diag(eigvals)
            @
            tf.transpose(
                eigvecs,
                perm=[0,2,1],
            )
        )


        # relative jitter
        scale = tf.reduce_mean(
            tf.linalg.diag_part(covs),
            axis=1,
            keepdims=True,
        )

        covs = covs + (
            tf.cast(1e-8, dtype)
            *
            tf.maximum(scale, 1.0)
            [:,:,None]
            *
            eye[None,:,:]
        )


        chol = tf.linalg.cholesky(covs)


        bad = tf.reduce_any(
            tf.math.is_nan(chol),
            axis=[1,2],
        )

        bad = tf.logical_and(
            bad,
            nodes_accept,
        )

        return (
            i + 1,
            covs,
            chol,
            bad,
        )


    _, covs, chol, bad = tf.while_loop(
        cond,
        body,
        (
            tf.constant(0),
            covs,
            chol,
            bad,
        ),
        maximum_iterations=max_iter,
    )


    return covs, chol



class ProgressPrinter:

    def __init__(self, term_width=None):

        self.is_notebook = (
            "ipykernel" in sys.modules
        )

        if self.is_notebook:
            from IPython.display import DisplayHandle, HTML
            self.HTML = HTML
            self.handle = DisplayHandle()
            self.handle.display("")
            if term_width is None:
                term_width = 50
        else:
            if term_width is None:
                try:
                    term_width = shutil.get_terminal_size().columns
                except:
                    term_width = 50
            self.n_lines = 0
        self.term_width = term_width

    def update(self, text):

        if self.is_notebook:
            html_text ="<pre>" + text + "</pre>"
            self.handle.update(self.HTML(html_text))

        else:

            if self.n_lines > 0:
                print(
                    "\033[F" * self.n_lines,
                    end=""
                )
            lines = text.split("\n")
            new_lines = []
            for line in lines:
                if len(line) > self.term_width:
                    line = line[:self.term_width-3] + "..."
                elif len(line) < self.term_width:
                    line = line + " " * (self.term_width - len(line))
                new_lines.append(line)
            text = "\n".join(new_lines)

            print(text, end="")

            self.n_lines = text.count("\n")

