import copy
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import CloughTocher2DInterpolator, CubicSpline
from scipy.optimize import minimize, brentq
from best import Sampler

class OptimiserResults():
    def __init__(self, vals, min_loglike, min_position, idxs, idx_reduced):
        self.fixed_points = vals
        self.loglkl = min_loglike
        self.reduced_position = min_position
        self.full_position = tf.transpose(
                tf.reshape(
                        tf.dynamic_stitch([idx_reduced, idxs],
                                          [tf.transpose(self.reduced_position),
                                           self.fixed_points]),
                        [self.reduced_position.shape[-1] + self.fixed_points.shape[0],
                         tf.shape(self.reduced_position)[0]]))
        self.idxs = idxs

class Optimiser:
    def __init__(self,
                 log_prob_fn,
                 bounds,
                 covmat=None,
                 loc=None,
                 mcmc_temperature=1.0):
        self.N_param_tot = len(bounds[0])
        lower, upper = bounds
        self.lower = tf.cast(lower, dtype=tf.float32)
        self.upper = tf.cast(upper, dtype=tf.float32)
        self.log_prob_fn, self.covmat, self.loc, self.samples = self.compute_covmat(log_prob_fn, covmat, loc, mcmc_temperature)

    def compute_covmat(self,
                       log_prob_fn,
                       covmat,
                       loc,
                       mcmc_temperature):
        s = Sampler(log_prob_fn, bounds=(self.lower, self.upper))
        print('Sampling parameter space')
        print('Sampling with AIES for 1000 steps and 500 chains')
        res = s.sample(method='aies', n_steps=1000, n_chains=500, initial_distribution='uniform', get_individual_chains=False, temperature=mcmc_temperature, verbose=False)
        best_sample = res.samples[tf.math.argmax(res.loglkl)]
        print('Sampling with HMC for 1000 steps and 50 chains with three burn-in iterations of 100 steps each')
        res = s.sample(method='hmc', n_steps=1000, n_chains=50, initial_state=best_sample, initial_distribution='repeat', get_individual_chains=False, temperature=mcmc_temperature)
        print('Sampling completed')
        if loc is None:
            loc = res.samples[tf.math.argmax(res.loglkl)]
        if covmat is None:
            covmat = tfp.stats.covariance(res.samples)
        loglike_fn = lambda x: -s.log_prob_fn(x)
        return loglike_fn, covmat, loc, res.samples

    def update_proposal(self,
                        temp,
                        loglike_mins,
                        positions,
                        tril_reduced):
        loglike_min = tf.reduce_min(loglike_mins, axis=1)
        indices = tf.math.argmin(loglike_mins, axis=1)
        nbins = indices.shape[0]
        positions_min = tf.stack([positions[i, indices[i], :] for i in range(nbins)], 0)

        return loglike_min, positions_min, [tfp.distributions.MultivariateNormalTriL(loc=positions_min[i],scale_tril=tril_reduced*tf.sqrt(temp)) for i in range(nbins)]

    def optimise_points(self,
                        points,
                        idxs,
                        batch_size=10,
                        step_size=0.05,
                        min_step_size=1e-5,
                        max_correct_loglike=10000,
                        start_temperature=1.0,
                        min_temperature=1e-2,
                        decay_temperature=0.5,
                        decay_step_size=0.5,
                        verbose=True):
        N_param = self.N_param_tot - len(idxs)

        nbins = points.shape[-1]
        vals = tf.repeat(points, repeats=batch_size, axis=1)

        idx_reduced = []
        for i in range(self.loc.shape[0]):
            if not tf.where(tf.equal(idxs, i)).shape[0]:
                idx_reduced.append(i)
        idx_reduced = tf.constant(idx_reduced, dtype=tf.int32)
        loc_reduced = tf.gather(self.loc, idx_reduced)
        lower_reduced = tf.gather(tf.constant(self.lower, dtype=tf.float32), idx_reduced)
        upper_reduced = tf.gather(tf.constant(self.upper, dtype=tf.float32), idx_reduced)

        cov_reduced = tf.gather(tf.gather(self.covmat, idx_reduced, axis=0), idx_reduced, axis=1)
        tril_reduced = tf.linalg.cholesky(cov_reduced)
        T=start_temperature
        prop_dist = [tfp.distributions.MultivariateNormalTriL(loc=loc_reduced,scale_tril=tril_reduced)]*nbins

        min_loglike = tf.constant([1e+10]*nbins)
        min_position = tf.zeros((nbins,N_param), dtype=tf.float32)
        while T > min_temperature:
            if verbose:
                print('Temperature is T =', T)
            y = tf.stack([prop_dist[i].sample((batch_size)) for i in range(nbins)])
            X = tf.clip_by_value(y, lower_reduced, upper_reduced)
            X = tf.reshape(X, [nbins*batch_size, N_param])
            position, loglike_val = self.optimise(X, cov_reduced, (lower_reduced, upper_reduced), vals, idx_reduced, idxs, n_steps=100, lr=step_size)
        
            while (tf.math.reduce_any(tf.math.is_nan(loglike_val)) or tf.math.reduce_any(loglike_val > max_correct_loglike)) and step_size > min_step_size:
                if verbose:
                    print('Bad loglike values found, trying again with lower step size.')
                step_size *= decay_step_size
                position, loglike_val = self.optimise(X, cov_reduced, (lower_reduced, upper_reduced), vals, idx_reduced, idxs, n_steps=100, lr=step_size)

            loglike_mins = tf.reshape(loglike_val, [nbins, batch_size])
            positions = tf.reshape(position, [nbins, batch_size, N_param])
            T *= decay_temperature
            loglike_min, bestfit, prop_dist = self.update_proposal(T, loglike_mins, positions, tril_reduced)

            idx2update = tf.where(loglike_min < min_loglike)[:,0]
            min_loglike = tf.tensor_scatter_nd_update(min_loglike, idx2update[:,None], tf.gather(loglike_min, idx2update))
            min_position = tf.tensor_scatter_nd_update(min_position, idx2update[:,None], tf.gather(bestfit, idx2update))
        return min_loglike, min_position

    @tf.function(jit_compile=True)
    def loglike(self,
                x,
                vals,
                idx_reduced,
                idxs):
        batch_size = tf.shape(x)[0]
        y = tf.transpose(
            tf.reshape(
                tf.dynamic_stitch([idx_reduced, idxs],
                                  [tf.transpose(x),
                                   vals]),
                [self.N_param_tot,
                 batch_size]))

        return self.log_prob_fn(y)

    @tf.function(jit_compile=True)
    def loglike_grad(self,
                     x,
                     vals,
                     idx_reduced,
                     idxs):
        return tfp.math.value_and_gradient(lambda x: self.loglike(x, vals, idx_reduced, idxs), x)

    @tf.function(jit_compile=True)
    def optimise(self,
                 x0,
                 prec,
                 bounds,
                 vals,
                 idx_reduced,
                 idxs,
                 n_steps=50,
                 lr=0.05):

        x = x0
        val = tf.zeros(
            tf.shape(x0)[:-1],
            dtype=x0.dtype
        )

        def body(i, x, val):

            val, g = self.loglike_grad(x, vals, idx_reduced, idxs)

            step = tf.einsum("...ij,...j->...i", prec, g)
            x = x - tf.expand_dims(lr, -1) * step

            x = tf.clip_by_value(x, bounds[0], bounds[1])

            return i + 1, x, val

        _, x_final, val_final = tf.while_loop(
            lambda i, x, val: i < n_steps,
            body,
            [0, x0, val]
        )

        return x_final, val_final

    def histogram2d(self,
                    samples,
                    centers_x,
                    centers_y):
        hist2d = []
        dx = centers_x[1] - centers_x[0]
        dy = centers_y[1] - centers_y[0]

        for i in range(len(centers_x)):
            limits = [centers_x[i]-dx/2, centers_x[i]+dx/2]
            s = tf.gather_nd(samples, tf.where(samples[:,0] >= limits[0]))
            s = tf.gather_nd(s, tf.where(s[:,0] < limits[1]))
            edges = tf.linspace(centers_y[0]-dy/2, centers_y[-1]+dy/2, len(centers_y)+1)
            hist = tfp.stats.histogram(s[:,1], edges=edges)
            hist2d.append(hist)
        hist2d = tf.stack(hist2d, axis=0)
        return hist2d

    def get_bins(self,
                 nbins=20,
                 idxs=[]):
        if len(idxs) == 0:
            idxs = list(range(self.samples.shape[-1]))
        bin_centers = []
        for i in idxs:
            edges = tf.linspace(self.lower[i], self.upper[i], nbins+1)
            hist = tfp.stats.histogram(self.samples[:,i], edges=edges)
            indices = tf.where(hist > 0)[:,0]
            first_index = tf.reduce_min(indices)
            last_index = tf.reduce_max(indices)
            count = 0
            while tf.cast(last_index - first_index + 1, tf.float32) < tf.cast(tf.reduce_min([0.9*nbins,nbins-3]), tf.float32):
                if count > 20:
                    break
                edges = tf.linspace(edges[tf.reduce_max([first_index-1, 0])], edges[tf.reduce_min([last_index+2, nbins])], nbins+1)
                try:
                    hist = tfp.stats.histogram(self.samples[:,i], edges=edges)
                except:
                    print(i, edges, first_index, last_index)
                    raise ValueError('Error in histogram calculation - check that the number of bins is not too high')
                indices = tf.where(hist > 0)[:,0]
                first_index = tf.reduce_min(indices)
                last_index = tf.reduce_max(indices)
                count += 1
            centers = (edges[:-1] + edges[1:]) / 2
            bin_centers.append(centers)
        bin_centers = tf.stack(bin_centers, axis=0)
        return bin_centers

    def get_2d_bins(self,
                    idx1,
                    idx2,
                    nbins=20):
        bin_centers = self.get_bins(nbins=nbins, idxs=[idx1, idx2])
        centers_x = bin_centers[0,:]
        centers_y = bin_centers[1,:]
        hist2d = self.histogram2d(tf.gather(self.samples, [idx1, idx2], axis=1), centers_x, centers_y)
        mask = tf.cast(hist2d, tf.bool)
        pairs = []
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i,j]:
                    pairs.append([bin_centers[0,i], bin_centers[1,j]])
        pairs = tf.transpose(tf.concat([pairs], axis=0))
        return pairs

    def compute_profile(self,
                        idxs=[],
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
                        verbose=True):
        if len(idxs) == self.N_param_tot:
            raise ValueError("All parameters are fixed - no optimisation can be performed. Please provide a subset of parameters to optimise over using the 'idxs' argument.")
        if fixed_points is not None and fixed_points.shape[0] == len(idxs):
            vals = tf.constant(fixed_points, dtype=tf.float32)
        elif len(idxs) == 0:
            print("Computing global minimum")
            vals = tf.zeros((0, 1), dtype=tf.float32)
        elif len(idxs) == 1:
            print("Computing 1D profile for parameter", idxs[0])
            vals = self.get_bins(nbins=nbins, idxs=idxs)
        elif len(idxs) == 2:
            print("Computing 2D profile for parameters", idxs[0], "and", idxs[1])
            vals = self.get_2d_bins(idxs[0], idxs[1], nbins=nbins)
        else:
            if nd_fixed is not None and nd_fixed.shape[0] == len(idxs):
                print(f"Computing {len(idxs)}d profile for parameters", idxs)
                vals = nd_fixed
            else:
                raise ValueError("Only global optimisation, 1D profiles, and 2D profiles can automatically be computed. For higher dimensions, please provide a set of points to minimise the log-likelihood at using the nd_fixed argument. The shape must be (n, m), where n is the number of fixed parameters (length of 'idxs') and m is the number of points to optimise.")
        min_loglike, min_position = self.optimise_points(vals, idxs, batch_size=batch_size, step_size=step_size, min_step_size=min_step_size, max_correct_loglike=max_correct_loglike, min_temperature=min_temperature, decay_temperature=decay_temperature, decay_step_size=decay_step_size, start_temperature=start_temperature, verbose=verbose)
        opt_res = OptimiserResults(vals, min_loglike, min_position, idxs, tf.constant([i for i in range(len(self.loc)) if i not in idxs], dtype=tf.int32))
        return opt_res

    def plot_profile_2d(self,
                        opt_res,
                        lkl_min_global=None,
                        ax=None,
                        contours=True):
        if opt_res.fixed_points.shape[0] != 2:
            raise ValueError("Only 2D profiles can be plotted. Please provide an OptimiserResults object with exactly 2 fixed parameters. For 1D profiles, use the plot_1d_profile method.")
        pi = opt_res.fixed_points[0]
        pj = opt_res.fixed_points[1]
        lkl = opt_res.loglkl
        if lkl_min_global is None:
            lkl_min_global = tf.reduce_min(opt_res.loglkl)
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        if contours:
            X = np.linspace(min(pi), max(pi))
            Y = np.linspace(min(pj), max(pj))
            X, Y = np.meshgrid(X, Y)
            interp = CloughTocher2DInterpolator(list(zip(pi, pj)), lkl, fill_value=100000, rescale=True)
            Z = interp(X, Y)
            c = 'r'
            ax.contour(X,Y,Z,levels=[lkl_min_global+11.83/2],zorder=4,colors=c,linewidths=3)
            ax.contour(X,Y,Z,levels=[lkl_min_global+6.18/2],zorder=4,colors=c,linewidths=3)
            ax.contour(X,Y,Z,levels=[lkl_min_global+2.3/2],zorder=4,colors=c,linewidths=3)

        ax.scatter(pi, pj, c=lkl, cmap='gist_rainbow', s=80)
        ax.set_xlabel(f'Parameter {opt_res.idxs[0]}')
        ax.set_ylabel(f'Parameter {opt_res.idxs[1]}')
        mappable = plt.cm.ScalarMappable(cmap='gist_rainbow')
        mappable.set_array(lkl)
        plt.colorbar(mappable, label=r'$-log \mathcal{L}$', ax=ax)
        plt.subplots_adjust(right=1.0)
        plt.show()

    def plot_profile_1d(self,
                        opt_res,
                        lkl_min_global=None,
                        ax=None,
                        confidence_intervals=True):
        if opt_res.fixed_points.shape[0] != 1:
            raise ValueError("Only 1D profiles can be plotted. Please provide an OptimiserResults object with exactly 1 fixed parameter. For 2D profiles, use the plot_2d_profile method.")
        p = opt_res.fixed_points[0].numpy()
        lkl = opt_res.loglkl.numpy()
        if lkl_min_global is None:
            lkl_min_global = tf.reduce_min(opt_res.loglkl)
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        ax.plot(p, lkl, 'o-', markersize=8, color='C0')
        ax.set_xlabel(f'Parameter {opt_res.idxs[0]}')
        ax.set_ylabel(r'$-log \mathcal{L}$')
        if confidence_intervals:
            self.compute_ci(ax, p, lkl, lkl_min_global)

        plt.show()

    def compute_ci(self,
                   ax,
                   p,
                   lkl,
                   lkl_min_global):

        p, lkl = zip(*sorted(zip(p, lkl)))

        spline = CubicSpline(p, lkl, bc_type='natural')

        try:
            res = minimize(
                fun=spline,
                x0=p[np.argmin(lkl)],
                method='Nelder-Mead',
                bounds=[(p[0], p[-1])]
            )
            x_min = res.x[0]
        except:
            return

        fn1 = lambda x: spline(x) - lkl_min_global - 1./2
        fn2 = lambda x: spline(x) - lkl_min_global - 4./2

        def safe_bracket(fn, a, b):
            try:
                return brentq(fn, a, b)
            except:
                return None

        x1_1 = safe_bracket(fn1, p[0], x_min)
        x1_2 = safe_bracket(fn1, x_min, p[-1])
        x2_1 = safe_bracket(fn2, p[0], x_min)
        x2_2 = safe_bracket(fn2, x_min, p[-1])

        x_extend = ax.get_xlim()[1] - ax.get_xlim()[0]
        y_extend = ax.get_ylim()[1] - ax.get_ylim()[0]

        def draw_line(x, label, alpha):

            if x is None:
                return

            ax.axvline(x, color='k', linestyle='--', linewidth=2, alpha=alpha)
            ax.text(
                x - 0.03 * x_extend,
                ax.get_ylim()[0] + 0.5 * y_extend,
                label,
                rotation=90,
                verticalalignment='bottom',
                color='k',
                alpha=alpha
            )

        draw_line(x1_1, r'1$\sigma$', 0.5)
        draw_line(x1_2, r'1$\sigma$', 0.5)
        draw_line(x2_1, r'2$\sigma$', 0.2)
        draw_line(x2_2, r'2$\sigma$', 0.2)

    def add_points_1d(self,
                      opt_res,
                      confidence_intervals=True,
                      lkl_min_global=None,
                      batch_size=10,
                      start_temperature=1.0,
                      decay_temperature=0.5,
                      min_temperature=1e-2,
                      step_size=0.05,
                      min_step_size=1e-5,
                      decay_step_size=0.5,
                      max_correct_loglike=10000):

        self.new_points = []
        result = {
            "opt_res": copy.deepcopy(opt_res),
            "closed": False
        }
        ci_flag = confidence_intervals
        fig, ax = plt.subplots()
        p = opt_res.fixed_points[0].numpy()
        lkl = opt_res.loglkl.numpy()

        # store vertical line artists
        vlines = []

        def redraw():
            ax.clear()
            current = result["opt_res"]
            p = current.fixed_points[0].numpy()
            lkl = current.loglike.numpy()
            ax.plot(p, lkl, 'o-', markersize=8, color='C0')
            ax.set_xlabel(f'Parameter {current.idxs[0]}')
            ax.set_ylabel(r'$-log \mathcal{L}$')

            if ci_flag:
                p = current.fixed_points[0].numpy()
                lkl = current.loglike.numpy()
                lkl_min = (
                    lkl_min_global
                    if lkl_min_global is not None
                    else np.min(lkl)
                )
                self.compute_ci(ax, p, lkl, lkl_min)

            # redraw vertical lines
            for x in self.new_points:
                ax.axvline(x, color='C1', linestyle='-', linewidth=2, alpha=0.8)

            fig.canvas.draw_idle()

        redraw()

        # -----------------------
        # CLICK: select new point
        # -----------------------
        def onclick(event):
            if result["closed"]:
                return
            if event.xdata is None:
                return
            self.new_points.append(event.xdata)
            redraw()

        # -----------------------
        # ENTER: compute new points
        # -----------------------
        def onkeypress(event):
            if result["closed"]:
                return
            if event.key != "enter":
                return
            if len(self.new_points) == 0:
                print("No selected points.")
                return
            print("Computing new points...")
            old = result["opt_res"]
            fixed = tf.constant(
                np.array(self.new_points, dtype=np.float32).reshape(1, -1),
                dtype=tf.float32
            )
            new = self.compute_profile(
                idxs=old.idxs,
                fixed_points=fixed,
                batch_size=batch_size,
                step_size=step_size,
                min_step_size=min_step_size,
                max_correct_loglike=max_correct_loglike,
                min_temperature=min_temperature,
                decay_temperature=decay_temperature,
                decay_step_size=decay_step_size,
                start_temperature=start_temperature
            )
            # sort results py parameter value
            p = tf.concat(
                [old.fixed_points, new.fixed_points],
                axis=1
            ).numpy()[0]
            lkl = tf.concat(
                [old.loglike, new.loglike],
                axis=0
            ).numpy()
            reduced_pos = tf.concat(
                [old.reduced_position, new.reduced_position],
                axis=0
            ).numpy()
            full_pos = tf.concat(
                [old.full_position, new.full_position],
                axis=0
            ).numpy()
            p, lkl, reduced_pos, full_pos = zip(*sorted(zip(p, lkl, reduced_pos, full_pos)))
            
            # append results
            old.fixed_points = tf.constant(tf.stack([p], axis=0), dtype=tf.float32)
            old.loglike = tf.constant(lkl, dtype=tf.float32)
            old.reduced_position = tf.constant(reduced_pos, dtype=tf.float32)
            old.full_position = tf.constant(full_pos, dtype=tf.float32)
            result["opt_res"] = old
            self.new_points = []

            redraw()

        def onclose(event):
            result["closed"] = True

        fig.canvas.mpl_connect("button_press_event", onclick)
        fig.canvas.mpl_connect("key_press_event", onkeypress)
        fig.canvas.mpl_connect("close_event", onclose)

        plt.show()

        return result["opt_res"]

    def recompute_points_1d(self,
                            opt_res,
                            confidence_intervals=True,
                            lkl_min_global=None,
                            batch_size=10,
                            start_temperature=1.0,
                            decay_temperature=0.5,
                            min_temperature=1e-2,
                            step_size=0.05,
                            min_step_size=1e-5,
                            decay_step_size=0.5,
                            max_correct_loglike=10000):

        self.selected_idx = set()
        result = {
            "opt_res": copy.deepcopy(opt_res),
            "closed": False
        }
        ci_flag = confidence_intervals
        fig, ax = plt.subplots()

        def redraw():
            ax.clear()
            current = result["opt_res"]
            p = current.fixed_points[0].numpy()
            lkl = current.loglike.numpy()
            ax.plot(p, lkl, 'o-', markersize=8, color='C0')
            # black overlay for selected points
            if len(self.selected_idx) > 0:
                idxs = list(self.selected_idx)
                ax.scatter(
                    p[idxs],
                    lkl[idxs],
                    c="black",
                    s=80,
                    zorder=5
                )

            ax.set_xlabel(f'Parameter {current.idxs[0]}')
            ax.set_ylabel(r'$-log \mathcal{L}$')

            if ci_flag:
                p = current.fixed_points[0].numpy()
                lkl = current.loglike.numpy()
                lkl_min = np.min(lkl)
                self.compute_ci(ax, p, lkl, lkl_min)

            fig.canvas.draw_idle()

        redraw()

        # -----------------------
        # CLICK: select nearest
        # -----------------------
        def onclick(event):
            if result["closed"]:
                return
            if event.xdata is None:
                return
            current = result["opt_res"]
            p = current.fixed_points[0].numpy()
            best_idx = None
            best_dist = np.inf
            for i, xi in enumerate(p):
                if i in self.selected_idx:
                    continue
                d = abs(xi - event.xdata)
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            if best_idx is None:
                return
            self.selected_idx.add(best_idx)

            redraw()

        # -----------------------
        # ENTER: recompute selected
        # -----------------------
        def onkeypress(event):
            if result["closed"]:
                return
            if event.key != "enter":
                return
            if len(self.selected_idx) == 0:
                print("No selected points.")
                return
            print("Recomputing selected points...")
            old = result["opt_res"]
            idxs = list(self.selected_idx)
            fixed = tf.gather(
                old.fixed_points,
                idxs,
                axis=1
            )
            recomputed = self.compute_profile(
                idxs=old.idxs,
                fixed_points=fixed,
                batch_size=batch_size,
                step_size=step_size,
                min_step_size=min_step_size,
                max_correct_loglike=max_correct_loglike,
                min_temperature=min_temperature,
                decay_temperature=decay_temperature,
                decay_step_size=decay_step_size,
                start_temperature=start_temperature
            )

            # convert to numpy for safe elementwise logic
            old_loglike_np = old.loglike.numpy()
            old_red_np = old.reduced_position.numpy()
            old_full_np = old.full_position.numpy()

            new_loglike_np = recomputed.loglike.numpy()
            new_red_np = recomputed.reduced_position.numpy()
            new_full_np = recomputed.full_position.numpy()

            mask = new_loglike_np < old_loglike_np[idxs]  # only accept improvements

            # if nothing improves, do nothing
            if not np.any(mask):
                print("No improvements found.")
                self.selected_idx = set()
                redraw()
                return

            for j, i in enumerate(idxs):

                if not mask[j]:
                    continue

                old_loglike_np[i] = new_loglike_np[j]
                old_red_np[i, :] = new_red_np[j, :]
                old_full_np[i, :] = new_full_np[j, :]

            old.loglike = tf.constant(old_loglike_np, dtype=old.loglike.dtype)
            old.reduced_position = tf.constant(old_red_np, dtype=old.reduced_position.dtype)
            old.full_position = tf.constant(old_full_np, dtype=old.full_position.dtype)

            result["opt_res"] = old
            self.selected_idx = set()

            redraw()

        def onclose(event):
            result["closed"] = True

        fig.canvas.mpl_connect("button_press_event", onclick)
        fig.canvas.mpl_connect("key_press_event", onkeypress)
        fig.canvas.mpl_connect("close_event", onclose)

        plt.show()

        return result["opt_res"]

    def add_points_2d(self,
                      opt_res,
                      contours=True,
                      lkl_min_global=None,
                      batch_size=10,
                      start_temperature=1.0,
                      decay_temperature=0.5,
                      min_temperature=1e-2,
                      step_size=0.05,
                      min_step_size=1e-5,
                      decay_step_size=0.5,
                      max_correct_loglike=10000):

        self.new_points = []
        result = {
            "opt_res": opt_res,
            "closed": False
        }
        fig, ax = plt.subplots()
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

        # -------------------------
        # FIXED COLOR SCALE SETUP
        # -------------------------
        lkl0 = opt_res.loglkl.numpy()
        if lkl_min_global is None:
            contour_min = {"value": np.min(opt_res.loglkl.numpy())}
        else:
            contour_min = {"value": lkl_min_global}
        vmin = np.min(lkl0)
        vmax = np.max(lkl0)
        base_range = vmax - vmin
        step = base_range / 20.0

        # current adjustable vmax (ONLY this changes)
        state = {
            "vmin": vmin,
            "vmax": vmax
        }
        norm = plt.Normalize(vmin=vmin, vmax=vmax, clip=True)
        cmap = plt.cm.gist_rainbow

        # -------------------------
        # COLORBAR (fixed instance)
        # -------------------------
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(lkl0)
        cbar = fig.colorbar(
            mappable,
            ax=ax,
            label=r'$-log \mathcal{L}$'
        )
        fig.subplots_adjust(right=1.0)
        marker = {"obj": None}

        # -------------------------
        # REDRAW FUNCTION
        # -------------------------
        def redraw():
            ax.clear()
            current = result["opt_res"]
            pi = current.fixed_points[0].numpy()
            pj = current.fixed_points[1].numpy()
            lkl = current.loglike.numpy()

            # clip automatically handled via norm.clip=True
            ax.scatter(
                pi,
                pj,
                c=lkl,
                cmap=cmap,
                norm=norm,
                s=80
            )

            # contours (fixed from initial scale)
            if contours and len(pi) >= 3:
                X = np.linspace(np.min(pi), np.max(pi))
                Y = np.linspace(np.min(pj), np.max(pj))
                X, Y = np.meshgrid(X, Y)
                interp = CloughTocher2DInterpolator(
                    list(zip(pi, pj)),
                    lkl,
                    fill_value=100000,
                    rescale=True
                )
                Z = interp(X, Y)
                base = contour_min["value"]
                ax.contour(
                    X, Y, Z,
                    levels=[
                        base + 2.3/2,
                        base + 6.18/2,
                        base + 11.83/2
                    ],
                    colors="r",
                    linewidths=3
                )
            ax.set_xlabel(f"Parameter {current.idxs[0]}")
            ax.set_ylabel(f"Parameter {current.idxs[1]}")
            ax.set_title(
                "Click to add points - Enter = optimise - ↑↓ adjust color scale"
            )
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            marker["obj"], = ax.plot([], [], "ko", markersize=10)

            fig.canvas.draw_idle()

        redraw()

        # -------------------------
        # CLICK HANDLER
        # -------------------------
        def onclick(event):
            if result["closed"]:
                return
            if event.xdata is None or event.ydata is None:
                return
            self.new_points.append((event.xdata, event.ydata))
            pts = np.array(self.new_points).T
            marker["obj"].set_data(pts[0], pts[1])

            fig.canvas.draw_idle()

        # -------------------------
        # APPLY COLOR SCALE UPDATE
        # -------------------------
        def update_color_scale():
            # enforce bounds: vmax cannot go below vmin + step
            if state["vmax"] < state["vmin"] + step:
                state["vmax"] = state["vmin"] + step
            norm.vmin = state["vmin"]
            norm.vmax = state["vmax"]
            mappable.set_norm(norm)
            cbar.update_normal(mappable)

            fig.canvas.draw_idle()

        # -------------------------
        # KEYBOARD HANDLER
        # -------------------------
        def onkeypress(event):
            if result["closed"]:
                return

            # -------------------------
            # OPTIMISATION
            # -------------------------
            if event.key == "enter":
                if len(self.new_points) == 0:
                    print("No new points.")
                    return
                print("Optimising new points...")
                old = result["opt_res"]
                new_points = tf.transpose(
                    tf.constant(self.new_points, dtype=tf.float32)
                )
                new = self.compute_profile(
                    idxs=old.idxs,
                    fixed_points=new_points,
                    step_size=step_size,
                    min_step_size=min_step_size,
                    max_correct_loglike=max_correct_loglike,
                    min_temperature=min_temperature,
                    decay_temperature=decay_temperature,
                    decay_step_size=decay_step_size,
                    start_temperature=start_temperature
                )
                # merge results
                new.fixed_points = tf.concat(
                    [old.fixed_points, new.fixed_points],
                    axis=1
                )
                new.loglike = tf.concat(
                    [old.loglike, new.loglike],
                    axis=0
                )
                new.reduced_position = tf.concat(
                    [old.reduced_position, new.reduced_position],
                    axis=0
                )
                new.full_position = tf.concat(
                    [old.full_position, new.full_position],
                    axis=0
                )
                if lkl_min_global is None:
                    new_min = tf.reduce_min(new.loglike).numpy()
                    contour_min["value"] = min(
                        contour_min["value"],
                        new_min
                    )
                    if new_min < state["vmin"]:
                        state["vmin"] = new_min
                        norm.vmin = state["vmin"]
                        mappable.set_norm(norm)
                        cbar.update_normal(mappable)

                result["opt_res"] = new
                self.new_points = []

                redraw()
                return

            # -------------------------
            # COLORBAR CONTROL
            # -------------------------
            if event.key == "up":
                state["vmax"] += step
                update_color_scale()
            elif event.key == "down":
                state["vmax"] -= step
                update_color_scale()

        # -------------------------
        # CLOSE HANDLER (safe exit)
        # -------------------------
        def onclose(event):
            result["closed"] = True

        # -------------------------
        # CONNECT EVENTS
        # -------------------------
        fig.canvas.mpl_connect("button_press_event", onclick)
        fig.canvas.mpl_connect("key_press_event", onkeypress)
        fig.canvas.mpl_connect("close_event", onclose)

        plt.show()

        return result["opt_res"]

    def recompute_points_2d(self,
                            opt_res,
                            contours=True,
                            lkl_min_global=None,
                            batch_size=10,
                            start_temperature=1.0,
                            decay_temperature=0.5,
                            min_temperature=1e-2,
                            step_size=0.05,
                            min_step_size=1e-5,
                            decay_step_size=0.5,
                            max_correct_loglike=10000):

        self.selected_idx = set()
        result = {
            "opt_res": copy.deepcopy(opt_res),
            "closed": False
        }
        fig, ax = plt.subplots()

        # -------------------------
        # COLORBAR SETUP
        # -------------------------

        lkl0 = opt_res.loglkl.numpy()
        if lkl_min_global is None:
            contour_min = {"value": np.min(opt_res.loglkl.numpy())}
        else:
            contour_min = {"value": lkl_min_global}
        vmin = np.min(lkl0)
        vmax = np.max(lkl0)
        step = (vmax - vmin) / 20.0
        color_state = {
            "vmax": vmax,
            "vmin": vmin
        }
        norm = plt.Normalize(
            vmin=vmin,
            vmax=vmax,
            clip=True
        )
        cmap = plt.cm.gist_rainbow
        mappable = plt.cm.ScalarMappable(
            norm=norm,
            cmap=cmap
        )
        mappable.set_array(lkl0)
        cbar = fig.colorbar(
            mappable,
            ax=ax,
            label=r'$-log \mathcal{L}$'
        )
        fig.subplots_adjust(right=1.0)
        marker = {"obj": None}

        # -------------------------
        # REDRAW
        # -------------------------

        def redraw():
            ax.clear()
            current = result["opt_res"]
            pi = current.fixed_points[0].numpy()
            pj = current.fixed_points[1].numpy()
            lkl = current.loglike.numpy()
            ax.scatter(
                pi,
                pj,
                c=lkl,
                cmap=cmap,
                norm=norm,
                s=80
            )
            # selected points
            if len(self.selected_idx) > 0:
                sel = list(self.selected_idx)
                ax.scatter(
                    pi[sel],
                    pj[sel],
                    c="black",
                    s=120,
                    zorder=5
                )
            # contours
            if contours and len(pi) >= 3:
                X = np.linspace(
                    np.min(pi),
                    np.max(pi)
                )
                Y = np.linspace(
                    np.min(pj),
                    np.max(pj)
                )
                X, Y = np.meshgrid(X, Y)
                interp = CloughTocher2DInterpolator(
                    list(zip(pi, pj)),
                    lkl,
                    fill_value=100000,
                    rescale=True
                )
                Z = interp(X, Y)
                base = contour_min["value"]
                ax.contour(
                    X,
                    Y,
                    Z,
                    levels=[
                        base + 2.3/2,
                        base + 6.18/2,
                        base + 11.83/2
                    ],
                    colors="r",
                    linewidths=3
                )
            ax.set_xlabel(
                f"Parameter {current.idxs[0]}"
            )
            ax.set_ylabel(
                f"Parameter {current.idxs[1]}"
            )
            ax.set_title(
                "Click to select points - Enter to recompute - ↑↓ adjust color scale"
            )
            ax.xaxis.set_major_locator(
                MaxNLocator(nbins=6)
            )
            ax.yaxis.set_major_locator(
                MaxNLocator(nbins=6)
            )
            marker["obj"], = ax.plot(
                [],
                [],
                "ko",
                markersize=10
            )
            fig.canvas.draw_idle()

        redraw()



        # -------------------------
        # SELECT POINTS
        # -------------------------
        def onclick(event):
            if result["closed"]:
                return
            if event.xdata is None or event.ydata is None:
                return
            current = result["opt_res"]
            pi = current.fixed_points[0].numpy()
            pj = current.fixed_points[1].numpy()
            best_idx = None
            best_dist = np.inf
            for i, (x, y) in enumerate(zip(pi, pj)):
                if i in self.selected_idx:
                    continue
                x_range = np.max(pi) - np.min(pi)
                y_range = np.max(pj) - np.min(pj)
                x_scale = x_range if x_range > 0 else 1.0
                y_scale = y_range if y_range > 0 else 1.0
                dx = (x - event.xdata) / x_scale
                dy = (y - event.ydata) / y_scale
                dist = dx * dx + dy * dy
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            if best_idx is None:
                return
            self.selected_idx.add(best_idx)

            redraw()

        # -------------------------
        # COLOR UPDATE
        # -------------------------
        def update_color():
            if color_state["vmax"] < color_state["vmin"] + step:
                color_state["vmax"] = (
                    color_state["vmin"] + step
                )
            norm.vmin = color_state["vmin"]
            norm.vmax = color_state["vmax"]
            mappable.set_norm(norm)
            cbar.update_normal(mappable)
            fig.canvas.draw_idle()

        # -------------------------
        # KEY HANDLER
        # -------------------------
        def onkeypress(event):
            if result["closed"]:
                return
            if event.key == "up":
                color_state["vmax"] += step
                update_color()
                return
            if event.key == "down":
                color_state["vmax"] -= step
                update_color()
                return
            if event.key != "enter":
                return
            if len(self.selected_idx) == 0:
                print("No selected points.")
                return
            print("Recomputing selected points...")
            old = result["opt_res"]
            idxs = list(self.selected_idx)
            selected_fixed = tf.gather(
                old.fixed_points,
                idxs,
                axis=1
            )
            recomputed = self.compute_profile(
                idxs=old.idxs,
                fixed_points=selected_fixed,
                step_size=step_size,
                min_step_size=min_step_size,
                max_correct_loglike=max_correct_loglike,
                min_temperature=min_temperature,
                decay_temperature=decay_temperature,
                decay_step_size=decay_step_size,
                start_temperature=start_temperature
            )

            # convert to numpy for safe comparison
            old_loglike_np = old.loglike.numpy()
            old_red_np = old.reduced_position.numpy()
            old_full_np = old.full_position.numpy()

            new_loglike_np = recomputed.loglike.numpy()
            new_red_np = recomputed.reduced_position.numpy()
            new_full_np = recomputed.full_position.numpy()

            mask = new_loglike_np < old_loglike_np[idxs]

            for j, i in enumerate(idxs):

                if not mask[j]:
                    continue

                old_loglike_np[i] = new_loglike_np[j]
                old_red_np[i, :] = new_red_np[j, :]
                old_full_np[i, :] = new_full_np[j, :]

            old.loglike = tf.constant(old_loglike_np, dtype=old.loglike.dtype)
            old.reduced_position = tf.constant(old_red_np, dtype=old.reduced_position.dtype)
            old.full_position = tf.constant(old_full_np, dtype=old.full_position.dtype)

            if lkl_min_global is None:
                new_min = tf.reduce_min(old.loglike).numpy()
                contour_min["value"] = min(
                    contour_min["value"],
                    new_min
                )
                if new_min < color_state["vmin"]:
                    color_state["vmin"] = new_min
                    norm.vmin = color_state["vmin"]
                    mappable.set_norm(norm)
                    cbar.update_normal(mappable)
            result["opt_res"] = old
            self.selected_idx = set()

            redraw()

        # -------------------------
        # CLOSE
        # -------------------------
        def onclose(event):
            result["closed"] = True

        fig.canvas.mpl_connect(
            "button_press_event",
            onclick
        )
        fig.canvas.mpl_connect(
            "key_press_event",
            onkeypress
        )
        fig.canvas.mpl_connect(
            "close_event",
            onclose
        )

        plt.show()

        return result["opt_res"]
