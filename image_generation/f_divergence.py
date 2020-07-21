import tensorflow as tf


def stable_exp(x):
    return tf.exp(tf.clip_by_value(x, clip_value_min=-5., clip_value_max=5.))


def get_divergence_funcs(divergence):
    conjugate_grad_exp = None
    grad_exp = None

    if divergence == 'reverse_kl':
        def grad_exp(x):
            return - stable_exp(-x)

        def conjugate_grad_exp(x):
            return -1. + x

    if divergence == 'kl':
        def grad_exp(x):
            return 1 + x

        def conjugate_grad_exp(x):
            return stable_exp(x)

    if divergence == 'pearson_x2':
        def grad_exp(x):
            return 2. * (stable_exp(x) - 1)

        def conjugate_grad_exp(x):
            return stable_exp(2. * x) - 1

    if divergence == 'jensen_shannon':
        def grad_exp(x):
            return tf.log(2.) + x - tf.log(1 + stable_exp(x))

        def conjugate_grad_exp(x):
            return -tf.log(2.) + tf.log(1 + stable_exp(x))

    if divergence == 'squared_hellinger':
        def grad_exp(x):
            return 1. - stable_exp(-0.5 * x)

        def conjugate_grad_exp(x):
            return stable_exp(0.5 * x) - 1

    return grad_exp, conjugate_grad_exp
