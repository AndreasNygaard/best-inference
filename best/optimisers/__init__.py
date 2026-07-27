import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


def GradientDescent(loglike,
                    loglike_grad,
                    x0,
                    prec,
                    bounds,
                    n_steps=50,
                    lr=0.05):

    x = x0
    val = tf.zeros(
        tf.shape(x0)[:-1],
        dtype=x0.dtype
    )

    def body(i, x, val):

        val, g = loglike_grad(x)

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


def GradientDescentLineSearch(loglike,
                              loglike_grad,
                              x0,
                              prec,
                              bounds,
                              n_steps=50,
                              lr=0.05,
                              armijo_c=1e-4,
                              backtrack=0.5,
                              max_line_search_steps=6):

    x = x0

    batch_shape = tf.shape(x0)[:-1]

    val = tf.zeros(batch_shape, dtype=x0.dtype)

    lr = tf.cast(lr, x0.dtype)
    armijo_c = tf.cast(armijo_c, x0.dtype)
    backtrack = tf.cast(backtrack, x0.dtype)

    def body(i, x, val):

        # Current objective and gradient
        val, g = loglike_grad(x)

        # Descent direction
        p = -tf.einsum("...ij,...j->...i", prec, g)

        gTp = tf.reduce_sum(g * p, axis=-1)

        alpha = tf.fill(tf.shape(val), lr)

        active = tf.ones(tf.shape(val), tf.bool)

        ls_iter = tf.constant(0)

        def ls_cond(ls_iter, alpha, active):
            return tf.logical_and(
                ls_iter < max_line_search_steps,
                tf.reduce_any(active)
            )

        def ls_body(ls_iter, alpha, active):

            x_trial = x + alpha[..., None] * p

            val_trial = loglike(x_trial)

            armijo_rhs = val + armijo_c * alpha * gTp

            accepted = val_trial <= armijo_rhs

            newly_accepted = tf.logical_and(active, accepted)

            # Freeze alpha for accepted points
            active = tf.logical_and(active, tf.logical_not(accepted))

            alpha = tf.where(
                active,
                backtrack * alpha,
                alpha
            )

            return ls_iter + 1, alpha, active

        _, alpha, _ = tf.while_loop(
            ls_cond,
            ls_body,
            [ls_iter, alpha, active]
        )

        x = x + alpha[..., None] * p

        return i + 1, x, val

    _, x_final, val_final = tf.while_loop(
        lambda i, *_: i < n_steps,
        body,
        [0, x, val]
    )

    return x_final, val_final


def DiagonalDFP(loglike,
                    loglike_grad,
                    x0,
                    prec,
                    bounds,
                    n_steps=50,
                    lr=0.1,
                    damping=1e-8,
                    momentum=0.8):

    x = x0
    v = tf.zeros_like(x)

    # Initial inverse Hessian approximation
    H = tf.ones_like(x0) * tf.linalg.diag_part(prec)

    val, g = loglike_grad(x)

    def body(i, x, g, val, v, H):

        # Search direction
        p = -H * g

        v = momentum * v + (1.0 - momentum) * p

        x_new = x + lr * v
        x_new = tf.clip_by_value(x_new, bounds[0], bounds[1])

        val_new, g_new = loglike_grad(x_new)

        s = x_new - x
        y = g_new - g

        # ---- Diagonal DFP update ----

        sty = tf.reduce_sum(s * y, axis=-1, keepdims=True)

        Hy = H * y
        yHy = tf.reduce_sum(y * Hy, axis=-1, keepdims=True)

        sty = tf.maximum(sty, damping)
        yHy = tf.maximum(yHy, damping)

        H = H \
            + tf.square(s) / sty \
            - tf.square(Hy) / yHy

        # Keep inverse Hessian positive
        H = tf.maximum(H, damping)

        return (
            i + 1,
            x_new,
            g_new,
            val_new,
            v,
            H
        )

    _, x_final, g_final, val_final, _, H = tf.while_loop(
        lambda i, *_: i < n_steps,
        body,
        [
            0,
            x,
            g,
            val,
            v,
            H
        ]
    )

    return x_final, val_final


def DiagonalBFGS(loglike,
                    loglike_grad,
                    x0,
                    prec,
                    bounds,
                    n_steps=50,
                    lr=0.1,
                    damping=1e-8,
                    momentum=0.8):

    x = x0
    v = tf.zeros_like(x)

    # Initial inverse Hessian approximation
    H = tf.ones_like(x0) * tf.linalg.diag_part(prec)

    val, g = loglike_grad(x)

    def body(i, x, g, val, v, H):

        # Search direction
        p = -H * g

        v = momentum * v + (1.0 - momentum) * p

        x_new = x + lr * v
        x_new = tf.clip_by_value(x_new, bounds[0], bounds[1])

        val_new, g_new = loglike_grad(x_new)

        s = x_new - x
        y = g_new - g

        # ---- Diagonal BFGS update ----

        sty = tf.reduce_sum(s * y, axis=-1, keepdims=True)
        sty = tf.maximum(sty, damping)

        rho = 1.0 / sty

        Hy = H * y
        yHy = tf.reduce_sum(y * Hy, axis=-1, keepdims=True)

        H = (
            H
            - 2.0 * rho * H * s * y
            + (rho * rho) * yHy * tf.square(s)
            + rho * tf.square(s)
        )

        # Keep inverse Hessian positive
        H = tf.maximum(H, damping)

        return (
            i + 1,
            x_new,
            g_new,
            val_new,
            v,
            H
        )

    _, x_final, g_final, val_final, _, H = tf.while_loop(
        lambda i, *_: i < n_steps,
        body,
        [
            0,
            x,
            g,
            val,
            v,
            H
        ]
    )

    return x_final, val_final


def DiagonalGaussNewton(loglike,
                        loglike_grad,
                        x0,
                        prec,
                        bounds,
                        n_steps=50,
                        lr=0.05,
                        damping=1e-3,
                        momentum=0.8):

    x = x0
    v = tf.zeros_like(x0)

    # running curvature estimate (diagonal Hessian proxy)
    P_diag = tf.linalg.diag_part(prec)
    gn_diag = tf.divide(tf.ones_like(x0), P_diag)  # initial curvature estimate

    beta = 0.8  # curvature EMA

    def body(i, x, v, gn_diag):

        val, g = loglike_grad(x)

        # ---- Gauss-Newton diagonal proxy ----
        # update curvature estimate from gradient magnitude
        empirical_gn_diag = tf.square(g)

        gn_diag = beta * gn_diag + (1.0 - beta) * empirical_gn_diag

        gn_diag_damped = gn_diag * (1.0 + damping)

        # Newton-like step
        p = -g / gn_diag_damped

        # optional momentum (usually 0 in tails, but harmless if small)
        v = momentum * v + (1.0 - momentum) * p

        x = x + lr * v

        # bounds
        x = tf.clip_by_value(x, bounds[0], bounds[1])

        return i + 1, x, v, gn_diag

    _, x_final, _, _ = tf.while_loop(
        lambda i, x, v, gn: i < n_steps,
        body,
        [0, x, v, gn_diag]
    )

    val_final = loglike(x_final)

    return x_final, val_final
