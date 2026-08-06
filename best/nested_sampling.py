import tensorflow as tf
import numpy as np
from scipy.stats import qmc

from best.tools import ProgressPrinter, safe_cholesky, slice_sampler, AdaptiveClusterTree

class NestedSamplerResults:

    def __init__(
        self,
        dead_points,
        dead_logL,
        live_points,
        live_logL,
        dead_logX,
        logZ,
        sigma_logZ,
        logX,
        H,
        log_weights,
        posterior_weights,
        n_live
    ):

        self.dead_points = dead_points
        self.dead_logL = dead_logL
        self.live_points = live_points
        self.live_logL = live_logL
        self.dead_logX = dead_logX
        self.logZ = logZ
        self.sigma_logZ = sigma_logZ
        self.logX = logX
        self.KLDivergence = H
        self.log_posterior_weights = log_weights
        self.posterior_weights = posterior_weights/tf.reduce_sum(posterior_weights)
        self.n_live = n_live

class NestedSampler:

    def __init__(
        self,
        log_prob_fn,
        bounds,
        n_live,
        n_live_updates=10,
        n_max_iter=100000,
        max_tree_depth=3,
        min_cluster_size=50,
        cluster_merge_tolerance=0.30,
        cluster_update_interval=100,
        slice_factor=5,
        slice_step_size=5.0,
        tolerance=1e-2,
        batch_sorting=True,
        history_correction=False,
        seed=42,
        dtype=tf.float32
    ):

        # --------------------------------------------------
        # User settings
        # --------------------------------------------------

        self.log_prob_fn = log_prob_fn

        self.n_iter = int(n_max_iter)
        self.n_live = int(n_live)
        self.n_live_updates = int(n_live_updates)

        self.max_tree_depth = int(max_tree_depth)
        self.min_cluster_size = int(min_cluster_size)

        self.cluster_merge_tolerance = float(cluster_merge_tolerance)
        self.cluster_update_interval = int(cluster_update_interval)

        self.slice_factor = int(slice_factor)
        self.slice_step_size = float(slice_step_size)

        self.tolerance = float(tolerance)
        self.batch_sorting = bool(batch_sorting)
        self.history_correction = bool(history_correction)
        self.seed = tf.constant([seed, seed+1], dtype=tf.int32)

        # --------------------------------------------------
        # Prior bounds
        # --------------------------------------------------

        lower, upper = bounds

        lower = np.asarray(lower)
        upper = np.asarray(upper)

        assert lower.ndim == 1
        assert upper.ndim == 1
        assert lower.shape == upper.shape

        self.ndim = lower.size
        self.dtype = dtype

        self.lower = tf.constant(lower, dtype=self.dtype)
        self.upper = tf.constant(upper, dtype=self.dtype)

        self.scales = self.upper - self.lower
        self.means = 0.5 * (self.upper + self.lower)

        self.scale_fn = lambda x: (x - self.means) / self.scales
        self.scale_fn_inv = lambda y: self.means + y * self.scales


        # --------------------------------------------------
        # Initial live points
        # --------------------------------------------------

        sampler = qmc.LatinHypercube(
            d=self.ndim,
            seed=seed,
        )

        lhs = sampler.random(self.n_live)

        lhs = lower + lhs * (upper - lower)

        self.live_points = self.scale_fn(
            tf.constant(
                lhs,
                dtype=self.dtype,
            )
        )
        
        self.live_logL = self.log_prob_fn_scaled(self.live_points)

        # select the best n_live points
        #best_indices = tf.argsort(live_logL, direction='DESCENDING')[:self.n_live]
        #self.live_points = tf.gather(live_points, best_indices)
        #self.live_logL = tf.gather(live_logL, best_indices)

        # --------------------------------------------------
        # Dead points
        # --------------------------------------------------

        self.dead_points = tf.zeros(
            (self.n_iter*self.n_live_updates, self.ndim),
            dtype=self.dtype,
        )

        self.dead_logL = tf.fill(
            (self.n_iter*self.n_live_updates,),
            tf.constant(-np.inf, self.dtype.as_numpy_dtype),
        )

        self.dead_logX = tf.fill(
            (self.n_iter*self.n_live_updates,),
            tf.constant(-np.inf, self.dtype.as_numpy_dtype),
        )

        # --------------------------------------------------
        # Evidence state
        # --------------------------------------------------

        self.logZ = tf.constant(
            -np.inf,
            self.dtype,
        )

        self.logZ_remaining = tf.constant(
            0.0,
            self.dtype,
        )

        self.logX = tf.constant(
            0.0,
            self.dtype,
        )

        self.H = tf.constant(
            0.0,
            self.dtype,
        )

        self.HZ = tf.constant(
            0.0,
            self.dtype,
        )

        # --------------------------------------------------
        # Iteration counter
        # --------------------------------------------------

        self.iteration = tf.constant(
            0,
            dtype=tf.int32,
        )

        # --------------------------------------------------
        # Clustering / ellipsoid sampler
        # --------------------------------------------------

        self.clusterer = AdaptiveClusterTree(
            max_depth=self.max_tree_depth,
            min_cluster_size=self.min_cluster_size,
            merge_tolerance=self.cluster_merge_tolerance,
        )

        self.slice_sampler = slice_sampler(
            loglike=self.log_prob_fn_scaled,
            bounds=(self.scale_fn(self.lower), self.scale_fn(self.upper)),
            n_iter=self.slice_factor*self.ndim,
            expand_to_worst=False,
        )

    def log_prob_fn_scaled(self, y):
        x = self.scale_fn_inv(y)
        return self.log_prob_fn(x)


    def logdiffexp(self, a, b):
        """
        Stable log(exp(a)-exp(b))
        requires a>b
        """
        return a + tf.math.log1p(
            -tf.exp(b-a)
        )


    @tf.function(jit_compile=True, reduce_retracing=True)
    def update_evidence(
        self,
        logZ,
        logX,
        dead_logLs,
        HZ
    ):

        k = tf.shape(dead_logLs)[0]

        steps = tf.cast(
            tf.range(k + 1),
            self.dtype
        )

        logXs = (
            logX
            -
            steps / tf.cast(self.n_live, self.dtype)
        )

        log_widths = self.logdiffexp(
            logXs[:-1],
            logXs[1:]
        )

        log_contribs = (
            dead_logLs
            +
            log_widths
        )

        # evidence update
        new_logZ = tf.reduce_logsumexp(
            tf.concat(
                [
                    [logZ],
                    log_contribs
                ],
                axis=0
            )
        )

        new_logX = logXs[-1]

        # Accumulate sum(z_i log L_i)
        HZ = HZ + tf.reduce_sum(
            tf.exp(log_contribs) * dead_logLs
        )

        H = HZ / tf.exp(new_logZ) - new_logZ


        return (
            new_logZ,
            new_logX,
            HZ,
            H
        )


    @tf.function(jit_compile=True, reduce_retracing=True)
    def run_iteration(
        self,
        state,
        batch_sorting=tf.constant(True, dtype=tf.bool),
        replace_history=tf.constant(False, dtype=tf.bool),
    ):

        (
            live_points,
            live_logL,
            dead_points,
            dead_logL,
            dead_logX,
            logZ,
            logX,
            logZ_remaining,
            H,
            iteration,
            nodes_accept,
            nodes_indices,
            nodes_volumes,
            nodes_chols,
            HZ
        ) = state


        # --------------------------------------------------
        # Find worst live points
        # --------------------------------------------------
        worst_indices = tf.nn.top_k(-live_logL, k=self.n_live_updates).indices
        worst_points = tf.gather(live_points, worst_indices)
        worst_logLs = tf.gather(live_logL, worst_indices)

        # --------------------------------------------------
        # Build clustering
        #
        # NOTE:
        # this assumes your clusterer is XLA compatible
        # and accepts tensors directly
        # --------------------------------------------------
        def run_clustering():

            (
                nodes_accept,
                nodes_indices,
                nodes_volumes,
                _,
                _,
                _,
                nodes_chols,
            ) = self.clusterer.get_all_clusters(
                live_points
            )

            # =====================================================
            # Recompute cluster covariances directly from live points
            # =====================================================

            w = tf.cast(
                nodes_indices,
                self.dtype,
            )  # (n_cluster, n_live)

            n = tf.reduce_sum(
                w,
                axis=1,
                keepdims=True,
            )  # (n_cluster, 1)

            live_points_batch = tf.repeat(live_points[None, :, :], repeats=tf.shape(nodes_indices)[0], axis=0)  # (n_cluster, n_live, ndim)

            means = (
                tf.reduce_sum(
                    live_points_batch * w[:, :, None],
                    axis=1,
                )
                / tf.maximum(n, 1.0)
            )

            deltas = (
                live_points_batch
                - means[:, None, :]
            )

            covs = tf.einsum(
                "nki,nkj->nij",
                deltas * w[:, :, None],
                deltas,
            )

            covs /= tf.maximum(
                n[:, :, None] - 1.0,
                1.0,
            )

            covs *= tf.cast(self.slice_step_size**2, self.dtype)  # inflate covariances slightly to avoid numerical issues

            # =====================================================
            # Repair covariance matrices + batched Cholesky
            # =====================================================

            nodes_covs, nodes_chols = safe_cholesky(
                covs,
                nodes_accept,
                self.ndim
            )

            # =====================================================
            # Verify Cholesky results for accepted clusters
            # =====================================================

            bad_chol = tf.reduce_any(
                tf.math.is_nan(nodes_chols),
                axis=[1,2],
            )

            bad_chol = tf.logical_and(
                bad_chol,
                nodes_accept,
            )


            # =====================================================
            # First fallback:
            # inflate diagonal of problematic covariances
            # =====================================================

            bad_mask = tf.cast(
                bad_chol[:,None,None],
                self.dtype,
            )

            diag = tf.linalg.diag_part(
                nodes_covs
            )

            inflated_diag = (
                diag
                *
                1.1
            )

            inflated_covs = tf.linalg.set_diag(
                nodes_covs,
                inflated_diag,
            )

            nodes_covs = (
                bad_mask * inflated_covs
                +
                (1.0 - bad_mask) * nodes_covs
            )


            # Try again

            nodes_covs, nodes_chols = safe_cholesky(
                nodes_covs,
                nodes_accept,
                self.ndim
            )


            # =====================================================
            # Second check
            # =====================================================

            bad_chol = tf.reduce_any(
                tf.math.is_nan(nodes_chols),
                axis=[1,2],
            )

            bad_chol = tf.logical_and(
                bad_chol,
                nodes_accept,
            )


            # =====================================================
            # Final fallback:
            # remove correlations
            # =====================================================

            bad_mask = tf.cast(
                bad_chol[:,None,None],
                self.dtype,
            )

            diag_covs = tf.linalg.diag(
                tf.linalg.diag_part(nodes_covs)
            )

            nodes_covs = (
                bad_mask * diag_covs
                +
                (1.0 - bad_mask) * nodes_covs
            )


            # Final Cholesky

            nodes_covs, nodes_chols = safe_cholesky(
                nodes_covs,
                nodes_accept,
                self.ndim
            )

            return (
                nodes_accept,
                nodes_indices,
                nodes_volumes,
                nodes_chols
            )
        (
            nodes_accept,
            nodes_indices,
            nodes_volumes,
            nodes_chols
        ) = tf.cond(
            tf.logical_or(
                tf.equal(iteration//self.n_live_updates % self.cluster_update_interval, 0),
                tf.less(iteration, 100)
            ),
            run_clustering,
            lambda: (
                nodes_accept,
                nodes_indices,
                nodes_volumes,
                nodes_chols
            )
        )

        # --------------------------------------------------
        # Find replacement point
        # --------------------------------------------------

        # start sampling from livepoints, but avoid the worst points that will be replaced
        alive_mask = tf.ones(
            (self.n_live,),
            dtype=tf.bool
        )
        alive_mask = tf.tensor_scatter_nd_update(
            alive_mask,
            worst_indices[:, None],
            tf.zeros_like(
                worst_indices,
                dtype=tf.bool
            )
        )
        weights = tf.cast(alive_mask, tf.float32)
        weights /= tf.reduce_sum(weights)

        idxs = tf.random.categorical(
            tf.math.log(weights[None, :]),
            self.n_live_updates
        )[0]

        x0s = tf.gather(live_points, idxs)

        # get corresponding cluster_id
        cluster_ids = tf.reduce_sum(
            tf.cast(
                tf.gather(nodes_indices, idxs, axis=1),
                tf.int32
            )
            *
            tf.expand_dims(
                tf.cast(
                    nodes_accept,
                    tf.int32
                ),
                axis=1
            )
            *
            tf.expand_dims(
                tf.range(
                    tf.shape(nodes_accept)[0],
                    dtype=tf.int32
                ),
                axis=1
            ),
            axis=0
        )

        Ls = tf.gather(nodes_chols, cluster_ids, axis=0)
        
        new_points, buffer_points, buffer_logLs = self.slice_sampler.sample(x0s, Ls, worst_logLs, seed=self.seed + tf.stack([iteration, iteration]))
        new_logLs = self.log_prob_fn_scaled(new_points)


        def no_sorting_fn():
            return worst_logLs, new_logLs, worst_points, new_points, tf.zeros((2*self.n_live_updates,), dtype=tf.int32)

        def batch_sorting_fn():
            combined_points = tf.concat(
                [worst_points, new_points],
                axis=0,
            )
            combined_logLs = tf.concat(
                [worst_logLs, new_logLs],
                axis=0,
            )

            # sort based on combined_logLs
            combined_logLs, sorted_indices = tf.nn.top_k(-combined_logLs, k=2*self.n_live_updates)
            combined_points = tf.gather(combined_points, sorted_indices)

            new_dead_points = combined_points[:self.n_live_updates]
            new_dead_logLs = -combined_logLs[:self.n_live_updates]
            new_live_points = combined_points[self.n_live_updates:]
            new_live_logLs = -combined_logLs[self.n_live_updates:]
            return new_dead_logLs, new_live_logLs, new_dead_points, new_live_points, sorted_indices

        new_dead_logLs, new_live_logLs, new_dead_points, new_live_points, sorted_indices = tf.cond(
            batch_sorting,
            batch_sorting_fn,
            no_sorting_fn
        )

        def only_sort_once_fn():
            return new_dead_logLs, new_live_logLs, new_dead_points, new_live_points

        def replace_history_fn():
            combined_seed_logLs = tf.concat(
                [-self.dtype.max * tf.ones_like(worst_logLs), worst_logLs],
                axis=0
            )
            combined_seed_logLs = tf.gather(combined_seed_logLs, sorted_indices)
            new_dead_seed_logLs = combined_seed_logLs[:self.n_live_updates]
            new_live_seed_logLs = combined_seed_logLs[self.n_live_updates:]

            invalid = new_live_seed_logLs >= new_live_logLs[0]

            relaxed = new_dead_seed_logLs > -self.dtype.max

            invalid_id = tf.cumsum(tf.cast(invalid, tf.int32), exclusive=True)

            invalid_id = tf.where(
                invalid,
                invalid_id,
                tf.fill([self.n_live_updates], -1)
            )

            relaxed_id = tf.cumsum(tf.cast(relaxed, tf.int32), exclusive=True)

            relaxed_id = tf.where(
                relaxed,
                relaxed_id,
                tf.fill([self.n_live_updates], -1)
            )

            ids = tf.range(self.n_live_updates)

            threshold_lookup = tf.reduce_max(
                tf.where(
                    ids[:, None] == relaxed_id[None, :],
                    new_dead_logLs[None, :],
                    tf.fill([self.n_live_updates, self.n_live_updates], -self.dtype.max),
                ),
                axis=1,
            )

            safe_invalid_id = tf.maximum(invalid_id,0)

            replacement_thresholds = tf.gather(
                threshold_lookup,
                safe_invalid_id
            )

            replacement_thresholds = tf.where(
                invalid,
                replacement_thresholds,
                tf.fill([self.n_live_updates], self.dtype.max)
            )

            seed_match = (
                new_live_seed_logLs[:,None]
                == worst_logLs[None,:]
            )

            buffer_indices = tf.argmax(
                tf.cast(seed_match,tf.int32),
                axis=1,
                output_type=tf.int32
            )

            history_logLs = tf.gather(
                buffer_logLs,
                buffer_indices,
                axis=1
            )

            history_logLs = tf.where(
                invalid[None,:],
                history_logLs,
                -self.dtype.max
            )

            history_points = tf.gather(
                buffer_points,
                buffer_indices,
                axis=1
            )

            history_points = tf.where(
                invalid[:,None],
                history_points,
                tf.ones_like(history_points)*(-self.dtype.max)
            )

            above = history_logLs > replacement_thresholds[None,:]

            prev = tf.concat(
                [tf.zeros_like(above[:1]), above[:-1]],
                axis=0,
            )

            first = tf.math.logical_and(
                above,
                tf.logical_not(prev)
            )

            history_idx = tf.argmax(
                tf.cast(first, tf.int32),
                axis=0,
                output_type=tf.int32,
            )

            cols = tf.range(self.n_live_updates, dtype=tf.int32)

            idx = tf.stack([history_idx, cols], axis=1)

            history_logLs_selected = tf.gather_nd(
                history_logLs,
                idx,
            )

            history_points_selected = tf.gather_nd(
                history_points,
                idx,
            )

            history_found = tf.reduce_any(first, axis=0)

            history_logLs_selected = tf.where(
                history_found,
                history_logLs_selected,
                tf.fill([self.n_live_updates], self.dtype.max),
            )

            history_points_selected = tf.where(
                history_found[:,None],
                history_points_selected,
                tf.ones_like(history_points_selected)*(-self.dtype.max),
            )

            actual_new_logLs = tf.where(
                invalid,
                history_logLs_selected,
                new_live_logLs,
            )

            actual_new_points = tf.where(
                invalid[:,None],
                history_points_selected,
                new_live_points,
            )

            combined_logLs = tf.concat(
                [new_dead_logLs, actual_new_logLs],
                axis=0,
            )

            combined_points = tf.concat(
                [new_dead_points, actual_new_points],
                axis=0,
            )

            order = tf.argsort(
                combined_logLs,
                direction="ASCENDING",
                stable=True,
            )

            combined_logLs = tf.gather(combined_logLs, order)
            combined_points = tf.gather(combined_points, order)

            actual_worst_logLs = combined_logLs[:self.n_live_updates]
            actual_new_logLs = combined_logLs[self.n_live_updates:]

            actual_worst_points = combined_points[:self.n_live_updates]
            actual_new_points = combined_points[self.n_live_updates:]

            return actual_worst_logLs, actual_new_logLs, actual_worst_points, actual_new_points


        actual_worst_logLs, actual_new_logLs, actual_worst_points, actual_new_points = tf.cond(
            tf.logical_and(
                replace_history,
                batch_sorting
            ),
            replace_history_fn,
            only_sort_once_fn
        )
        
        # --------------------------------------------------
        # Store dead point
        # --------------------------------------------------

        rows = (
            iteration
            +
            tf.range(self.n_live_updates)
        )

        dead_points = tf.tensor_scatter_nd_update(
            dead_points,
            rows[:,None],
            actual_worst_points,
        )

        dead_logL = tf.tensor_scatter_nd_update(
            dead_logL,
            rows[:,None],
            actual_worst_logLs,
        )


        # --------------------------------------------------
        # Replace live point
        # --------------------------------------------------

        live_points = tf.tensor_scatter_nd_update(
            live_points,
            worst_indices[:,None],
            actual_new_points,
        )

        live_logL = tf.tensor_scatter_nd_update(
            live_logL,
            worst_indices[:,None],
            actual_new_logLs,
        )


        # --------------------------------------------------
        # Evidence update
        # --------------------------------------------------

        logZ, logX, HZ, H = self.update_evidence(
            logZ,
            logX,
            actual_worst_logLs,
            HZ
        )

        logZ_remaining = (
            logX
            +
            tf.reduce_logsumexp(live_logL)
            - tf.math.log(
                tf.cast(self.n_live, self.dtype)
            )
        )

        steps = tf.cast(
            tf.range(
                self.n_live_updates,
                0,
                -1
            ),
            self.dtype
        )

        dead_logXs = (
            logX
            +
            (steps - 1.0)
            /
            self.n_live
        )

        dead_logX = tf.tensor_scatter_nd_update(
            dead_logX,
            rows[:,None],
            dead_logXs,
        )


        iteration = iteration + self.n_live_updates

        return (
            live_points,
            live_logL,
            dead_points,
            dead_logL,
            dead_logX,
            logZ,
            logX,
            logZ_remaining,
            H,
            iteration,
            nodes_accept,
            nodes_indices,
            nodes_volumes,
            nodes_chols,
            HZ
        )
    
    @tf.function
    def compute_diagnostics(self, state, lower, upper, dim_idx=0, bins=50):
        live_points = self.scale_fn_inv(state[0])
        #max_distance = tf.reduce_max(
        #    tf.norm(
        #        live_points[:, None, :] - live_points[None, :, :],
        #        axis=-1
        #    )
        #) # scales very badly
        center = tf.reduce_mean(live_points, axis=0)
        max_distance = tf.reduce_max(
            tf.norm(live_points - center, axis=1)
        ) # scales better, though approximate

        live_logL = state[1]
        max_logL = tf.reduce_max(live_logL)
        min_logL = tf.reduce_min(live_logL)

        # histogram of live points along the specified dimension
        hist = tf.histogram_fixed_width(
            live_points[:, dim_idx],
            value_range=(lower, upper),
            nbins=bins
        )

        return max_distance, max_logL, min_logL, hist
    
    @tf.function(jit_compile=True, reduce_retracing=True)
    def compute_posterior_weights(self, dead_logL, dead_logX, logZ):

        previous_logX = tf.concat(
            [
                tf.zeros((1,), dtype=self.dtype),
                dead_logX[:-1]
            ],
            axis=0
        )

        log_widths = self.logdiffexp(
            previous_logX,
            dead_logX,
        )

        log_weights = (
            dead_logL
            +
            log_widths
            -
            logZ
        )

        weights = tf.exp(log_weights)

        return log_weights, weights

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compute_final_logZ(self, state):
        logZ = state[5]
        logZ_remaining = state[7]
        final_logZ = tf.reduce_logsumexp(
            tf.stack(
                [
                    logZ,
                    logZ_remaining
                ],
                axis=0
            )
        )
        return final_logZ
    
    @tf.function(jit_compile=True, reduce_retracing=True)
    def run_chunk(
            self,
            state,
            n_iter,
            batch_sorting=tf.constant(True, dtype=tf.bool),
            replace_history=tf.constant(False, dtype=tf.bool),
    ):
        def cond(i, state):
            return tf.logical_not(
                tf.logical_or(
                    i > n_iter,
                    state[7] - state[5] < self.tolerance
                )
            )
        def body(i, state):
            state = self.run_iteration(state, replace_history=replace_history)
            return i + 1, state
        i, new_state = tf.while_loop(
            cond,
            body,
            (tf.constant(0), state),
        )
        return new_state, n_iter-i

    def print_progress(self, new_state, initial_max_distance, printer, lower=None, upper=None, dim_idx=0, final=False, final_sigma=None):
        term_width = printer.term_width
        max_distance, max_logL, min_logL, hist = self.compute_diagnostics(new_state, lower, upper, dim_idx=dim_idx, bins=term_width)
        logZ = new_state[5]
        sigma = tf.sqrt(new_state[8] / tf.cast(self.n_live, self.dtype))
        print_block = "╔" + "═"*(term_width - 2) + "╗\n"
        print_block += "║ Live cloud size" + " "*(term_width - 18) + "║\n"
        print_block += "╠" + "═"*(term_width - 2) + "╣\n"
        print_block += "║ " + "█"*(int(np.ceil((max_distance/initial_max_distance).numpy()*(term_width - 4)))) + "-"*(term_width - 4 - int(np.ceil((max_distance/initial_max_distance).numpy()*(term_width - 4)))) + " ║\n"
        print_block += "╚" + "═"*(term_width - 2) + "╝\n\n"
        print_block += f"LogZ          : {logZ.numpy():.4f} ± "
        if tf.math.is_nan(sigma):
            print_block += "------\n"
        else:
            print_block += f"{sigma.numpy():.4f}\n"
        print_block += f"LogZ_remain   : {new_state[7].numpy():.4f}\n"
        print_block += f"LogX          : {new_state[6].numpy():.4f}\n"
        print_block += f"Max logL_live : {max_logL.numpy():.4f}\n"
        print_block += f"∆logL_live    : {(max_logL-min_logL).numpy():.4f}\n"
        print_block += f"Clusters      : {tf.reduce_sum(tf.cast(new_state[10], tf.int32)).numpy()}\n"
        print_block += "\n\n"

        print_block += f"Distribution of live points along dimension {dim_idx}:\n\n"

        height = term_width // 10
        max_count = tf.reduce_max(hist)
        hist_scaled = tf.cast(tf.cast(hist, tf.float32) / tf.cast(max_count, tf.float32) * tf.cast(height, tf.float32), tf.int32)
        for i in range(height):
            for j in range(term_width):
                if hist_scaled[j] >= height - i:
                    print_block += "█"
                else:
                    if i == height - 1:
                        print_block += "_"
                    else:
                        print_block += " "
            print_block += "\n"
        print_block += "\n"
        lower_str = f"{lower.numpy():.3f}"
        upper_str = f"{upper.numpy():.3f}"
        print_block += lower_str + " "*(term_width - len(lower_str) - len(upper_str)) + upper_str + "\n"

        if final:
            logZ = self.compute_final_logZ(new_state)
            sigma = final_sigma
            print_block += "\n"
            print_block += f"Final LogZ : {logZ.numpy():.4f} ± "
            if tf.math.is_nan(sigma):
                print_block += "------\n"
            else:
                print_block += f"{sigma.numpy():.4f}\n"


        # find lowest index in hist with non-zero count
        non_zero_indices = tf.where(hist > 0)
        min_zero_idx = tf.cast(tf.reduce_min(non_zero_indices), tf.float32)
        max_zero_idx = tf.cast(tf.reduce_max(non_zero_indices), tf.float32)
        if min_zero_idx > term_width // 4:
            new_lower = lower + (upper - lower) * ((min_zero_idx-1) / tf.cast(term_width, tf.float32))
        else:
            new_lower = lower
        if max_zero_idx < 3 * term_width // 4:
            new_upper = lower + (upper - lower) * ((max_zero_idx+1) / tf.cast(term_width, tf.float32))
        else:
            new_upper = upper

        printer.update(print_block)

        return new_lower, new_upper

    def run(
            self,
            update_interval=10,
            display_param_idx=0,
            output_width=None,
            verbose=True,
            batch_sorting=None,
            history_correction=None
    ):
        n_cluster = 2**(self.max_tree_depth + 1) - 1
        new_state = (
            self.live_points,
            self.live_logL,
            self.dead_points,
            self.dead_logL,
            self.dead_logX,
            self.logZ,
            self.logX,
            self.logZ_remaining,
            self.H,
            self.iteration,
            tf.zeros((n_cluster,), dtype=tf.bool),
            tf.zeros((n_cluster, self.n_live), dtype=tf.bool),
            tf.zeros((n_cluster,), dtype=self.dtype),
            tf.zeros((n_cluster, self.ndim, self.ndim), dtype=self.dtype),
            self.HZ
        )

        if verbose:
            printer = ProgressPrinter(term_width=output_width)
            printer.update("Starting sampler...\n")

        if batch_sorting is None:
            batch_sorting = self.batch_sorting

        batch_sorting_tf = tf.logical_and(
            tf.constant(batch_sorting, dtype=tf.bool),
            self.n_live_updates > 1
        )

        if history_correction is None:
            history_correction = self.history_correction

        if self.slice_sampler.expand_to_worst != history_correction:
            self.slice_sampler = slice_sampler(
                loglike=self.log_prob_fn_scaled,
                bounds=(self.scale_fn(self.lower), self.scale_fn(self.upper)),
                n_iter=self.slice_factor*self.ndim,
                expand_to_worst=history_correction,
            )

        history_correction_tf = tf.logical_and(
            tf.constant(history_correction, dtype=tf.bool),
            self.n_live_updates > 1
        )

        if history_correction_tf and not batch_sorting_tf:
            warning_msg = "History correction is ignored because batch sorting is disabled. Please enable batch sorting to use history correction."
            printer.update(warning_msg + "\n")

        lower = self.lower[display_param_idx]
        upper = self.upper[display_param_idx]

        n_chunks = self.n_iter // update_interval
        n_iter_chunk = update_interval
        n_remainder = self.n_iter % update_interval

        # compile the run_chunk function for better performance
        #_ = self.run_chunk(new_state, n_iter_chunk)
        #_ = self.run_chunk(new_state, n_remainder)

        initial_max_distance, _, _, _ = self.compute_diagnostics(new_state, lower, upper, dim_idx=display_param_idx)

        for i in range(n_chunks):
            logZ_old = new_state[5]
            new_state, n_skipped = self.run_chunk(new_state, n_iter_chunk, batch_sorting=batch_sorting_tf, replace_history=history_correction_tf)
            if verbose:
                lower, upper = self.print_progress(new_state, initial_max_distance, printer, lower=lower, upper=upper, dim_idx=display_param_idx)
            logZ_new = new_state[5]
            logZ_remaining = new_state[7]
            if n_skipped > 0:
                break
        last_idx = ((i+1)*n_iter_chunk - n_skipped)*self.n_live_updates
        if i == n_chunks - 1:
            logZ_old = new_state[5]
            new_state, n_skipped = self.run_chunk(new_state, n_remainder, batch_sorting=batch_sorting_tf, replace_history=history_correction_tf)
            if verbose:
                _, _ = self.print_progress(new_state, initial_max_distance, printer, lower=lower, upper=upper, dim_idx=display_param_idx)
            logZ_new = new_state[5]
            logZ_remaining = new_state[7]
            if n_skipped == 0 or n_remainder == 0:
                print(f"Maximum number of iterations ({self.n_iter}) reached. Consider increasing n_max_iter")
            last_idx += (n_remainder - n_skipped)*self.n_live_updates
        
        dead_points = self.scale_fn_inv(new_state[2][:last_idx])
        live_points = self.scale_fn_inv(new_state[0])
        dead_logL = new_state[3][:last_idx]
        live_logL = new_state[1]
        logZ_final = self.compute_final_logZ(new_state)
        log_weights, weights = self.compute_posterior_weights(new_state[3], new_state[4], new_state[5])
        posterior_weights = weights[:last_idx] / tf.reduce_sum(weights[:last_idx])
        H = tf.reduce_sum(
            posterior_weights * tf.where(
                tf.math.is_finite(dead_logL),
                dead_logL,
                tf.zeros_like(dead_logL)
            )
        ) - logZ_final
        sigma = tf.sqrt(H / tf.cast(self.n_live, self.dtype))

        if verbose:
            _, _ = self.print_progress(new_state, initial_max_distance, printer, lower=lower, upper=upper, dim_idx=display_param_idx, final=True, final_sigma=sigma)

        results = NestedSamplerResults(
            dead_points,
            dead_logL,
            live_points,
            live_logL,
            new_state[4][:last_idx],
            logZ_final,
            sigma,
            new_state[6],
            H,
            log_weights[:last_idx],
            weights[:last_idx],
            self.n_live
        )
        return results 
