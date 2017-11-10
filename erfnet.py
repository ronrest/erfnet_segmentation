from __future__ import print_function, division, unicode_literals
import numpy as np
import tensorflow as tf


# USEFUL LAYERS
fc = tf.contrib.layers.fully_connected
conv = tf.contrib.layers.conv2d
# convsep = tf.contrib.layers.separable_conv2d
deconv = tf.contrib.layers.conv2d_transpose
relu = tf.nn.relu
maxpool = tf.contrib.layers.max_pool2d
dropout_layer = tf.layers.dropout
batchnorm = tf.contrib.layers.batch_norm
winit = tf.contrib.layers.xavier_initializer()
repeat = tf.contrib.layers.repeat
arg_scope = tf.contrib.framework.arg_scope
l2_regularizer = tf.contrib.layers.l2_regularizer


def get_conv_arg_scope(is_training, bn=True, reg=None, use_deconv=False, use_relu=True, bn_decay=0.9):
    with arg_scope(
        [deconv if use_deconv else conv],
        padding = "SAME",
        stride = 1,
        activation_fn = relu if use_relu else None,
        normalizer_fn = batchnorm if bn else None,
        normalizer_params = {"is_training": is_training, "decay": bn_decay},
        weights_regularizer = reg,
        variables_collections = None,
        ) as scope:
        return scope


    with arg_scope(get_conv_arg_scope(is_training=is_training, bn=True)):
def factorized_res_module(x, is_training, dropout=0.3, dilation=[1,1], l2=None, name="fres"):
    reg = None if l2 is None else l2_regularizer(l2)
    with arg_scope(get_conv_arg_scope(reg=reg, is_training=is_training, bn=True)):
        with tf.variable_scope(name):
            n_filters = x.shape.as_list()[-1]
            y = conv(x, num_outputs=n_filters, kernel_size=[3,1], rate=dilation[0], normalizer_fn=None, scope="conv_a_3x1")
            y = conv(y, num_outputs=n_filters, kernel_size=[1,3], rate=dilation[0], scope="conv_a_1x3")
            y = conv(y, num_outputs=n_filters, kernel_size=[3,1], rate=dilation[1], normalizer_fn=None, scope="conv_b_3x1")
            y = conv(y, num_outputs=n_filters, kernel_size=[1,3], rate=dilation[1], scope="conv_b_1x3")
            y = dropout_layer(y, rate=dropout)
            y = tf.add(x,y, name="add")
    print("DEBUG: {} {}".format(name, y.shape.as_list()))
    print("DEBUG: L2 in factorized res module {}".format(l2))
    return y


def downsample(x, n_filters, is_training, bn=False, use_relu=False, l2=None, name="down"):
    with tf.variable_scope(name):
        reg = None if l2 is None else l2_regularizer(l2)
        with arg_scope(get_conv_arg_scope(reg=reg, is_training=is_training, bn=bn, use_relu=use_relu)):
            n_filters_in = x.shape.as_list()[-1]
            n_filters_conv = n_filters - n_filters_in
            branch_a = conv(x, num_outputs=n_filters_conv, kernel_size=3, stride=2, scope="conv")
            branch_b = maxpool(x, kernel_size=2, stride=2, padding='VALID', scope="maxpool")
            y = tf.concat([branch_a, branch_b], axis=-1, name="concat")
    print("DEBUG: {} {}".format(name, y.shape.as_list()))
    return y

def upsample(x, n_filters, is_training=False, use_relu=False, bn=False, l2=None, name="up"):
    reg = None if l2 is None else l2_regularizer(l2)
    with arg_scope(get_conv_arg_scope(reg=reg, is_training=is_training, bn=bn, use_deconv=True, use_relu=use_relu)):
        y = deconv(x, num_outputs=n_filters, kernel_size=4, stride=2, scope=name)
    print("DEBUG: {} {}".format(name, y.shape.as_list()))
    return y


def erfnetA(X, Y, n_classes, alpha=0.001, dropout=0.3, l2=None, is_training=False):
    """
    """
    # TODO: Use TF repeat for some of these repeating layers
    # TODO: register factorized_res_module and upsample and downsample with arg_scope
    #       and pass dropout, is_training, etc to it.
    # TODO: Add weight decay.
    with tf.name_scope("preprocess") as scope:
        x = tf.div(X, 255., name="rescaled_inputs")

    x = downsample(x, n_filters=16, is_training=is_training, name="d1")

    x = downsample(x, n_filters=64, is_training=is_training, name="d2")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 1], name="fres3")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 1], name="fres4")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 1], name="fres5")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 1], name="fres6")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 1], name="fres7")

    # TODO: Use dilated convolutions
    x = downsample(x, n_filters=128, is_training=is_training, name="d8")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 2], name="fres9")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 4], name="fres10")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 8], name="fres11")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 16], name="fres12")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 2], name="fres13")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 4], name="fres14")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 8], name="fres15")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 16], name="fres16")

    x = upsample(x, n_filters=64, is_training=is_training, name="up17")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 1], name="fres18")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 1], name="fres19")

    x = upsample(x, n_filters=16, is_training=is_training, name="up20")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 1], name="fres21")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 1], name="fres22")

    x = upsample(x, n_filters=n_classes, is_training=is_training, name="up23")
    return x
def erfnetB(X, Y, n_classes, alpha=0.001, dropout=0.3, l2=None, is_training=False):
    """
    Uses L2 regularization.
    """
    print("DEBUG: L2 passed on to ERFNETB{}".format(l2))
    # TODO: Use TF repeat for some of these repeating layers
    # TODO: register factorized_res_module and upsample and downsample with arg_scope
    #       and pass dropout, is_training, etc to it.
    # TODO: Add weight decay.
    with tf.name_scope("preprocess") as scope:
        x = tf.div(X, 255., name="rescaled_inputs")

    x = downsample(x, n_filters=16, is_training=is_training, l2=l2, name="d1")

    x = downsample(x, n_filters=64, is_training=is_training, l2=l2, name="d2")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 1], l2=l2, name="fres3")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 1], l2=l2, name="fres4")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 1], l2=l2, name="fres5")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 1], l2=l2, name="fres6")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 1], l2=l2, name="fres7")

    # TODO: Use dilated convolutions
    x = downsample(x, n_filters=128, is_training=is_training, l2=l2, name="d8")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 2], l2=l2, name="fres9")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 4], l2=l2, name="fres10")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 8], l2=l2, name="fres11")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 16], l2=l2, name="fres12")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 2], l2=l2, name="fres13")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 4], l2=l2, name="fres14")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 8], l2=l2, name="fres15")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 16], l2=l2, name="fres16")

    x = upsample(x, n_filters=64, is_training=is_training, l2=l2, name="up17")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 1], l2=l2, name="fres18")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 1], l2=l2, name="fres19")

    x = upsample(x, n_filters=16, is_training=is_training, l2=l2, name="up20")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 1], l2=l2, name="fres21")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 1], l2=l2, name="fres22")

    x = upsample(x, n_filters=n_classes, is_training=is_training, l2=l2, name="up23")
    return x
