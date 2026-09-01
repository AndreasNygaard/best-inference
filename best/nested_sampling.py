import tensorflow as tf
import numpy as np

from best.tools import ProgressPrinter, safe_cholesky, slice_sampler, AdaptiveClusterTree

class NestedSamplerResults:

    def __init__(
        self,
        dead_points,
        dead_logL,
        live_points,
        live_logL,
        all_points,
        all_logL,
        dead_logX,
        logZ,
        sigma_logZ,
        logX,
        H,
        log_weights,
        posterior_weights,
        n_live,
    ):

        self.dead_points = dead_points
        self.dead_logL = dead_logL
        self.live_points = live_points
        self.live_logL = live_logL
        self.all_points = all_points
        self.all_logL = all_logL
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
        max_tree_depth=0,
        min_cluster_size=50,
        cluster_merge_tolerance=0.30,
        cluster_update_interval=100,
        slice_factor=5,
        slice_step_size=5.0,
        slice_global_mixing=0.1,
        tolerance=1e-3,
        batch_sorting=True,
        history_correction=True,
        history_correction_iterations=1,
        history_buffer_size=100,
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
        self.slice_global_mixing = float(slice_global_mixing)

        self.tolerance = float(tolerance)
        self.batch_sorting = bool(batch_sorting)
        self.history_correction = bool(history_correction)
        self.history_correction_iterations = int(history_correction_iterations)
        self.history_buffer_size = int(history_buffer_size)
        self.seed = tf.constant([seed, seed+1], dtype=tf.int32)

        # --------------------------------------------------
        # Prior bounds
        # --------------------------------------------------

        lower, upper = bounds

        self.lower_np = np.asarray(lower)
        self.upper_np = np.asarray(upper)

        assert self.lower_np.ndim == 1
        assert self.upper_np.ndim == 1
        assert self.lower_np.shape == self.upper_np.shape

        self.ndim = self.lower_np.size
        self.dtype = dtype

        self.lower = tf.constant(self.lower_np, dtype=self.dtype)
        self.upper = tf.constant(self.upper_np, dtype=self.dtype)

        self.scales = self.upper - self.lower
        self.means = 0.5 * (self.upper + self.lower)

        self.scale_fn = lambda x: (x - self.means) / self.scales
        self.scale_fn_inv = lambda y: self.means + y * self.scales

        self.initialise(seed=seed)


    def initialise(self, seed=42):
        # --------------------------------------------------
        # Initial live points
        # --------------------------------------------------

        # self.nlive uniform points between 0 and 1 in self.ndim dimensions
        points = np.random.uniform(
            low=self.lower_np,
            high=self.upper_np,
            size=(self.n_live, self.ndim)
        )

        self.live_points = self.scale_fn(
            tf.constant(
                points,
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
            expand_to_worst=self.history_correction,
            buffer_size=self.history_buffer_size,
            global_mixing=self.slice_global_mixing,
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
            HZ,
            nodes_centres,
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
                nodes_centres,
                _,
                _,
                nodes_chols,
            ) = tf.cond(
                tf.greater(self.max_tree_depth, 0),
                lambda: self.clusterer.get_all_clusters(
                    live_points
                ),
                lambda: (
                    tf.constant([True], dtype=tf.bool),
                    tf.ones((1, self.n_live), dtype=tf.bool),
                    tf.constant([1.0], dtype=self.dtype),
                    tf.reduce_mean(live_points, axis=0, keepdims=True),
                    tf.zeros((1, self.ndim, self.ndim), dtype=self.dtype),
                    tf.zeros((1, self.ndim, self.ndim), dtype=self.dtype),
                    tf.zeros((1, self.ndim, self.ndim), dtype=self.dtype),
                )
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
                nodes_chols,
                nodes_centres
            )
        (
            nodes_accept,
            nodes_indices,
            nodes_volumes,
            nodes_chols,
            nodes_centres
        ) = tf.cond(
            tf.logical_or(
                tf.equal(iteration % self.cluster_update_interval, 0),
                tf.less(iteration, 100)
            ),
            run_clustering,
            lambda: (
                nodes_accept,
                nodes_indices,
                nodes_volumes,
                nodes_chols,
                nodes_centres
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
        # the first cluster is the root cluster containing all points. Its covariance is the global covariance
        global_L = tf.gather(nodes_chols, 0, axis=0)

        new_points, buffer_points, buffer_logLs = self.slice_sampler.sample(x0s, Ls, global_L, worst_logLs, seed=self.seed + tf.stack([iteration, iteration]))
        new_logLs = self.log_prob_fn_scaled(new_points)

        # -------------------------------------------------------------------------
        # Batch sorting
        # -------------------------------------------------------------------------

        def no_sorting_fn():
            # No sorting -> no history correction.
            return (
                worst_logLs,
                new_logLs,
                worst_points,
                new_points,
                tf.zeros((2 * self.n_live_updates,), dtype=tf.int32),
            )

        def batch_sorting_fn():

            combined_points = tf.concat(
                [worst_points, new_points],
                axis=0,
            )

            combined_logLs = tf.concat(
                [worst_logLs, new_logLs],
                axis=0,
            )

            # Sort from worst -> best.
            combined_logLs, sorted_indices = tf.nn.top_k(
                -combined_logLs,
                k=2 * self.n_live_updates,
            )

            combined_logLs = -combined_logLs
            combined_points = tf.gather(
                combined_points,
                sorted_indices,
            )

            new_dead_points = combined_points[:self.n_live_updates]
            new_dead_logLs = combined_logLs[:self.n_live_updates]

            new_live_points = combined_points[self.n_live_updates:]
            new_live_logLs = combined_logLs[self.n_live_updates:]

            return (
                new_dead_logLs,
                new_live_logLs,
                new_dead_points,
                new_live_points,
                sorted_indices,
            )

        (
            new_dead_logLs,
            new_live_logLs,
            new_dead_points,
            new_live_points,
            sorted_indices,
        ) = tf.cond(
            batch_sorting,
            batch_sorting_fn,
            no_sorting_fn,
        )


        # -------------------------------------------------------------------------
        # History information for the newly generated points
        #
        # Each new point initially has:
        #
        #   parent_logLs = likelihood threshold used to generate it
        #
        #   history_indices = column in buffer_points/buffer_logLs corresponding
        #                     to its slice-sampling history.
        #
        # The original worst points do not have a history and are assigned -inf.
        # -------------------------------------------------------------------------

        # Before sorting, the first n_live_updates points are the old worst
        # points, while the second half are the newly generated points.
        original_parent_logLs = tf.concat(
            [
                -self.dtype.max * tf.ones_like(worst_logLs),
                worst_logLs,
            ],
            axis=0,
        )

        original_history_indices = tf.concat(
            [
                -tf.ones(
                    [self.n_live_updates],
                    dtype=tf.int32,
                ),
                tf.range(
                    self.n_live_updates,
                    dtype=tf.int32,
                ),
            ],
            axis=0,
        )

        parent_logLs_combined = tf.gather(
            original_parent_logLs,
            sorted_indices,
        )

        history_indices_combined = tf.gather(
            original_history_indices,
            sorted_indices,
        )

        new_dead_parent_logLs = parent_logLs_combined[
            :self.n_live_updates
        ]

        parent_logLs = parent_logLs_combined[
            self.n_live_updates:
        ]

        new_dead_history_indices = history_indices_combined[
            :self.n_live_updates
        ]

        history_indices = history_indices_combined[
            self.n_live_updates:
        ]


        # -------------------------------------------------------------------------
        # Iterative history correction
        # -------------------------------------------------------------------------

        def history_correction_step(
                dead_logLs,
                live_logLs,
                dead_points,
                live_points,
                dead_parent_logLs,
                live_parent_logLs,
                dead_history_indices,
                live_history_indices,
        ):

            # -------------------------------------------------------------
            # A live point is problematic if its parent threshold is >=
            # the worst current live likelihood.
            #
            # Its parent is therefore still alive and the point was generated
            # under a threshold which is no longer represented by a dead point.
            # -------------------------------------------------------------

            invalid = live_parent_logLs >= live_logLs[0]

            # -------------------------------------------------------------
            # Find relaxed thresholds.
            #
            # A dead point with a finite parent threshold represents a point
            # that was generated during this batch and subsequently became
            # an instant death / revived point.
            # -------------------------------------------------------------

            relaxed = dead_parent_logLs > -self.dtype.max

            relaxed_id = tf.cumsum(
                tf.cast(relaxed, tf.int32),
                exclusive=True,
            )

            relaxed_id = tf.where(
                relaxed,
                relaxed_id,
                tf.fill(
                    [self.n_live_updates],
                    -1,
                ),
            )

            ids = tf.range(
                self.n_live_updates,
                dtype=tf.int32,
            )

            # For each relaxed point, obtain the likelihood of the corresponding
            # dead point. These are the relaxed likelihood thresholds.
            threshold_lookup = tf.reduce_max(
                tf.where(
                    ids[:, None] == relaxed_id[None, :],
                    dead_logLs[None, :],
                    tf.fill(
                        [self.n_live_updates, self.n_live_updates],
                        -self.dtype.max,
                    ),
                ),
                axis=1,
            )

            # Invalid live points are matched, in order, to the relaxed dead
            # thresholds.
            invalid_id = tf.cumsum(
                tf.cast(invalid, tf.int32),
                exclusive=True,
            )

            invalid_id = tf.where(
                invalid,
                invalid_id,
                tf.fill(
                    [self.n_live_updates],
                    -1,
                ),
            )

            safe_invalid_id = tf.maximum(
                invalid_id,
                0,
            )
    
            replacement_thresholds = tf.gather(
                threshold_lookup,
                safe_invalid_id,
            )

            replacement_thresholds = tf.where(
                invalid,
                replacement_thresholds,
                tf.fill(
                    [self.n_live_updates],
                    self.dtype.max,
                ),
            )

            # -------------------------------------------------------------
            # Retrieve the history corresponding to each live point.
            #
            # history_indices tells us which original slice-sampling
            # trajectory to use.
            # -------------------------------------------------------------

            safe_history_indices = tf.maximum(
                live_history_indices,
                0,
            )

            history_logLs = tf.gather(
                buffer_logLs,
                safe_history_indices,
                axis=1,
            )

            history_points = tf.gather(
                buffer_points,
                safe_history_indices,
                axis=1,
            )

            history_logLs = tf.where(
                invalid[None, :],
                history_logLs,
                tf.fill(
                    tf.shape(history_logLs),
                    -self.dtype.max,
                ),
            )

            history_points = tf.where(
                invalid[None, :, None],
                history_points,
                tf.ones_like(history_points) * (-self.dtype.max),
            )

            # -------------------------------------------------------------
            # Find the first history point above the relaxed threshold.
            # -------------------------------------------------------------

            above = (
                history_logLs
                > replacement_thresholds[None, :]
            )

            prev = tf.concat(
                [
                    tf.zeros_like(above[:1]),
                    above[:-1],
                ],
                axis=0,
            )

            first = tf.logical_and(
                above,
                tf.logical_not(prev),
            )

            history_idx = tf.argmax(
                tf.cast(first, tf.int32),
                axis=0,
                output_type=tf.int32,
            )

            cols = tf.range(
                self.n_live_updates,
                dtype=tf.int32,
            )

            idx = tf.stack(
                [history_idx, cols],
                axis=1,
            )

            history_logLs_selected = tf.gather_nd(
                history_logLs,
                idx,
            )

            history_points_selected = tf.gather_nd(
                history_points,
                idx,
            )

            history_found = tf.reduce_any(
                first,
                axis=0,
            )

            history_logLs_selected = tf.where(
                history_found,
                history_logLs_selected,
                tf.fill(
                    [self.n_live_updates],
                    self.dtype.max,
                ),
            )

            history_points_selected = tf.where(
                history_found[:, None],
                history_points_selected,
                tf.ones_like(history_points_selected)
                * (-self.dtype.max),
            )

            # -------------------------------------------------------------
            # Replace problematic live points.
            # -------------------------------------------------------------

            actual_live_logLs = tf.where(
                invalid,
                history_logLs_selected,
                live_logLs,
            )

            actual_live_points = tf.where(
                invalid[:, None],
                history_points_selected,
                live_points,
            )

            # The important part for iterative correction:
            #
            # A corrected point was generated using the relaxed threshold,
            # so its NEW parent is that relaxed threshold.
            #
            # Its history index remains the same because we are reusing the
            # existing slice-sampling history rather than doing new sampling.
            # -------------------------------------------------------------

            actual_parent_logLs = tf.where(
                invalid,
                replacement_thresholds,
                live_parent_logLs,
            )

            actual_history_indices = live_history_indices

            # -------------------------------------------------------------
            # Sort dead + live points again.
            #
            # Parent information and history indices must be sorted together
            # with their associated points.
            # -------------------------------------------------------------

            combined_logLs = tf.concat(
                [
                    dead_logLs,
                    actual_live_logLs,
                ],
                axis=0,
            )

            combined_points = tf.concat(
                [
                    dead_points,
                    actual_live_points,
                ],
                axis=0,
            )

            combined_parent_logLs = tf.concat(
                [
                    dead_parent_logLs,
                    actual_parent_logLs,
                ],
                axis=0,
            )

            combined_history_indices = tf.concat(
                [
                    dead_history_indices,
                    actual_history_indices,
                ],
                axis=0,
            )

            order = tf.argsort(
                combined_logLs,
                direction="ASCENDING",
                stable=True,
            )

            combined_logLs = tf.gather(
                combined_logLs,
                order,
            )

            combined_points = tf.gather(
                combined_points,
                order,
            )

            combined_parent_logLs = tf.gather(
                combined_parent_logLs,
                order,
            )

            combined_history_indices = tf.gather(
                combined_history_indices,
                order,
            )

            actual_dead_logLs = combined_logLs[
                :self.n_live_updates
            ]

            actual_live_logLs = combined_logLs[
                self.n_live_updates:
            ]

            actual_dead_points = combined_points[
                :self.n_live_updates
            ]

            actual_live_points = combined_points[
                self.n_live_updates:
            ]

            actual_dead_parent_logLs = combined_parent_logLs[
                :self.n_live_updates
            ]

            actual_live_parent_logLs = combined_parent_logLs[
                self.n_live_updates:
            ]

            actual_dead_history_indices = combined_history_indices[
                :self.n_live_updates
            ]

            actual_live_history_indices = combined_history_indices[
                self.n_live_updates:
            ]

            # -------------------------------------------------------------
            # Determine whether another correction pass is required.
            # -------------------------------------------------------------

            still_invalid = (
                actual_live_parent_logLs
                >= actual_live_logLs[0]
            )

            any_invalid = tf.reduce_any(
                still_invalid
            )

            return (
                actual_dead_logLs,
                actual_live_logLs,
                actual_dead_points,
                actual_live_points,
                actual_dead_parent_logLs,
                actual_live_parent_logLs,
                actual_dead_history_indices,
                actual_live_history_indices,
                any_invalid,
            )


        # -------------------------------------------------------------------------
        # Run history correction repeatedly.
        # -------------------------------------------------------------------------

        max_history_corrections = self.history_correction_iterations

        def correction_cond(
                iteration,
                dead_logLs,
                live_logLs,
                dead_points,
                live_points,
                dead_parent_logLs,
                live_parent_logLs,
                dead_history_indices,
                live_history_indices,
                any_invalid,
        ):

            return tf.logical_and(
                iteration < max_history_corrections,
                any_invalid,
            )

        def correction_body(
                iteration,
                dead_logLs,
                live_logLs,
                dead_points,
                live_points,
                dead_parent_logLs,
                live_parent_logLs,
                dead_history_indices,
                live_history_indices,
                any_invalid,
        ):

            (
                dead_logLs,
                live_logLs,
                dead_points,
                live_points,
                dead_parent_logLs,
                live_parent_logLs,
                dead_history_indices,
                live_history_indices,
                any_invalid,
            ) = history_correction_step(
                dead_logLs,
                live_logLs,
                dead_points,
                live_points,
                dead_parent_logLs,
                live_parent_logLs,
                dead_history_indices,
                live_history_indices,
            )

            return (
                iteration + 1,
                dead_logLs,
                live_logLs,
                dead_points,
                live_points,
                dead_parent_logLs,
                live_parent_logLs,
                dead_history_indices,
                live_history_indices,
                any_invalid,
            )

        # If history correction is disabled, just return the once-sorted result.
        #
        # Otherwise perform up to history_correction_iterations passes.
        def do_history_correction():

            # Determine whether there is anything to correct initially.
            initial_invalid = (
                parent_logLs >= new_live_logLs[0]
            )

            initial_any_invalid = tf.reduce_any(
                initial_invalid
            )

            (
                _,
                final_dead_logLs,
                final_live_logLs,
                final_dead_points,
                final_live_points,
                final_dead_parent_logLs,
                final_live_parent_logLs,
                final_dead_history_indices,
                final_live_history_indices,
                _,
            ) = tf.while_loop(
                correction_cond,
                correction_body,
                (
                    tf.constant(0, dtype=tf.int32),
                    new_dead_logLs,
                    new_live_logLs,
                    new_dead_points,
                    new_live_points,
                    new_dead_parent_logLs,
                    parent_logLs,
                    new_dead_history_indices,
                    history_indices,
                    initial_any_invalid,
                ),
            )

            return (
                final_dead_logLs,
                final_live_logLs,
                final_dead_points,
                final_live_points,
            )

        def only_sort_once_fn():
            return (
                new_dead_logLs,
                new_live_logLs,
                new_dead_points,
                new_live_points,
            )

        (
            actual_worst_logLs,
            actual_new_logLs,
            actual_worst_points,
            actual_new_points,
        ) = tf.cond(
            tf.logical_and(
                replace_history,
                batch_sorting,
            ),
            do_history_correction,
            only_sort_once_fn,
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
        # Identify clusters for new points
        # --------------------------------------------------

        distances = tf.norm(
            actual_new_points[:, None, :] - nodes_centres[None, :, :],
            axis=-1
        )

        # set distances to inf for clusters that are not accepted
        distances = tf.where(
            nodes_accept[None, :],
            distances,
            tf.fill(
                tf.shape(distances),
                tf.constant(self.dtype.max, self.dtype)
            )
        )

        closest_cluster_ids = tf.cast(tf.argmin(
            distances,
            axis=1
        ), tf.int32)

        nodes_indices = tf.tensor_scatter_nd_update(
            nodes_indices,
            tf.stack([
                cluster_ids,
                worst_indices
            ], axis=1),
            tf.zeros((self.n_live_updates,), dtype=tf.bool)
        )

        nodes_indices = tf.tensor_scatter_nd_update(
            nodes_indices,
            tf.stack([
                closest_cluster_ids,
                worst_indices
            ], axis=1),
            tf.ones((self.n_live_updates,), dtype=tf.bool)
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

        n_nodes = 2 ** (self.max_tree_depth + 1) - 1
        nodes_accept = tf.ensure_shape(nodes_accept, (n_nodes,))
        nodes_indices = tf.ensure_shape(nodes_indices, (n_nodes, self.n_live))
        nodes_volumes = tf.ensure_shape(nodes_volumes, (n_nodes,))
        nodes_chols = tf.ensure_shape(nodes_chols, (n_nodes, self.ndim, self.ndim))
        nodes_centres = tf.ensure_shape(nodes_centres, (n_nodes, self.ndim))

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
            HZ,
            nodes_centres,
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
    def compute_posterior_weights(self, dead_logL, dead_logX, live_logL, logX, logZ):

        # Dead point weights
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

        dead_log_weights = (
            dead_logL
            +
            log_widths
            -
            logZ
        )

        live_log_weights = (
            live_logL
            +
            logX
            -
            tf.math.log(tf.cast(self.n_live, tf.float32))
            -
            logZ
        )

        # Combine
        log_weights = tf.concat(
            [
                dead_log_weights,
                live_log_weights
            ],
            axis=0
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
                    i >= n_iter,
                    state[7] - state[5] < self.tolerance
                )
            )
        def body(i, state):
            state = self.run_iteration(state, batch_sorting=batch_sorting, replace_history=replace_history)
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
            update_interval=100,
            display_param_idx=0,
            output_width=None,
            verbose=True,
            batch_sorting=None,
            history_correction=None,
            history_correction_iterations=None,
            history_buffer_size=None,
            seed=None,
    ):
        if seed is not None:
            self.seed = tf.constant([seed, seed+1], dtype=tf.int32)
            self.initialise(seed=seed)
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
            tf.constant(0, dtype=tf.int32),
            tf.zeros((n_cluster,), dtype=tf.bool),
            tf.zeros((n_cluster, self.n_live), dtype=tf.bool),
            tf.zeros((n_cluster,), dtype=self.dtype),
            tf.zeros((n_cluster, self.ndim, self.ndim), dtype=self.dtype),
            self.HZ,
            tf.zeros((n_cluster, self.ndim), dtype=self.dtype),
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

        if history_buffer_size is None:
            history_buffer_size = self.history_buffer_size

        if self.slice_sampler.expand_to_worst != history_correction:
            self.slice_sampler = slice_sampler(
                loglike=self.log_prob_fn_scaled,
                bounds=(self.scale_fn(self.lower), self.scale_fn(self.upper)),
                n_iter=self.slice_factor*self.ndim,
                expand_to_worst=history_correction,
                buffer_size=history_buffer_size,
            )

        history_correction_tf = tf.logical_and(
            tf.constant(history_correction, dtype=tf.bool),
            self.n_live_updates > 1
        )

        if history_correction_tf and not batch_sorting_tf:
            warning_msg = "History correction is ignored because batch sorting is disabled. Please enable batch sorting to use history correction."
            printer.update(warning_msg + "\n")

        if history_correction_iterations is not None:
            self.history_correction_iterations = history_correction_iterations

        lower = self.lower[display_param_idx]
        upper = self.upper[display_param_idx]

        n_chunks = self.n_iter // update_interval
        n_iter_chunk = update_interval
        n_remainder = self.n_iter % update_interval

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
        log_weights, weights = self.compute_posterior_weights(new_state[3], new_state[4], new_state[1], new_state[6], logZ_final)
        weights = tf.concat([weights[:last_idx], weights[-self.n_live:]], axis=0)
        log_weights = tf.concat([log_weights[:last_idx], log_weights[-self.n_live:]], axis=0)
        posterior_weights = weights / tf.reduce_sum(weights)
        all_points = tf.concat([dead_points, live_points], axis=0)
        all_logL = tf.concat([dead_logL, live_logL], axis=0)
        H = tf.reduce_sum(
            posterior_weights * tf.where(
                tf.math.is_finite(all_logL),
                all_logL,
                tf.zeros_like(all_logL)
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
            all_points,
            all_logL,
            new_state[4][:last_idx],
            logZ_final,
            sigma,
            new_state[6],
            H,
            log_weights,
            weights,
            self.n_live,
        )
        return results
