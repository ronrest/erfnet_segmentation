from __future__ import print_function, division, unicode_literals
import numpy as np
import tensorflow as tf

# from model_base import ImageClassificationModel
# from model_base import PretrainedImageClassificationModel
from data_processing import prepare_data
from model_base import SegmentationModel
import tensorflow as tf

__author__ = "Ronny Restrepo"
__copyright__ = "Copyright 2017, Ronny Restrepo"
__credits__ = ["Ronny Restrepo"]
__license__ = "Apache License"
__version__ = "2.0"


# ##############################################################################
#                                                                  ARCHITECTURES
# ##############################################################################
# USEFUL LAYERS
fc = tf.contrib.layers.fully_connected
conv = tf.contrib.layers.conv2d
# convsep = tf.contrib.layers.separable_conv2d
deconv = tf.contrib.layers.conv2d_transpose
relu = tf.nn.relu
maxpool = tf.contrib.layers.max_pool2d
dropout_layer = tf.layers.dropout
batchnorm = tf.contrib.layers.batch_norm
# bn_params = {"is_training": self.is_training}
winit = tf.contrib.layers.xavier_initializer()
repeat = tf.contrib.layers.repeat
arg_scope = tf.contrib.framework.arg_scope
l2_regularizer = tf.contrib.layers.l2_regularizer


# ==============================================================================
#                                                               GET_VGG_ARGSCOPE
# ==============================================================================
def get_vgg_argscope(weight_decay=0.0005, use_batch_norm=False, is_training=False):
    """ Gets the arg scope needed for VGG.
    Args:
        weight_decay: The l2 regularization coefficient.
    Returns:
        An arg_scope with the default arguments for layers in VGG .
    """
    with tf.contrib.framework.arg_scope(
        [conv],
        activation_fn=tf.nn.relu,
        normalizer_fn = batchnorm if use_batch_norm else None,
        normalizer_params = {"is_training": is_training},
        weights_regularizer=l2_regularizer(weight_decay),
        biases_initializer=tf.zeros_initializer(),
        trainable = True):
        with tf.contrib.framework.arg_scope([conv], padding='SAME') as scope:
                return scope


# ==============================================================================
#                                                                   VGG16_TRUNK
# ==============================================================================
def vgg16_trunk(inputs, weight_decay=0.0005, use_batch_norm=False, is_training=False):
    """ VGG layers before the fully connected layers """
    with tf.variable_scope("vgg_16", "vgg_16"):
        with tf.contrib.framework.arg_scope(get_vgg_argscope(
                weight_decay=weight_decay,
                use_batch_norm=use_batch_norm,
                is_training=is_training)):
            endpoints = {}
            x = repeat(inputs, 2, conv, num_outputs=64, kernel_size=3, scope='conv1')
            endpoints["conv1"] = x
            x = maxpool(x, kernel_size=2, scope='pool1')
            endpoints["pool1"] = x
            x = repeat(x, 2, conv, num_outputs=128, kernel_size=3, scope='conv2')
            endpoints["conv2"] = x
            x = maxpool(x, kernel_size=2, scope='pool2')
            endpoints["pool2"] = x
            x = repeat(x, 3, conv, num_outputs=256, kernel_size=3, scope='conv3')
            endpoints["conv3"] = x
            x = maxpool(x, kernel_size=2, scope='pool3')
            endpoints["pool3"] = x
            x = repeat(x, 3, conv, num_outputs=512, kernel_size=3, scope='conv4')
            endpoints["conv4"] = x
            x = maxpool(x, kernel_size=2, scope='pool4')
            endpoints["pool4"] = x
            x = repeat(x, 3, conv, num_outputs=512, kernel_size=3, scope='conv5')
            endpoints["conv5"] = x
            x = maxpool(x, kernel_size=2, scope='pool5')
            endpoints["pool5"] = x
            return x, endpoints


# ##############################################################################
#                                                                   AUGMENTATION
# ##############################################################################
# from image_processing import create_augmentation_func
#
# aug_func = create_augmentation_func(
#     shadow=(0.01, 0.8),
#     shadow_file="shadow_pattern.jpg",
#     shadow_crop_range=(0.02, 0.5),
#     rotate=30,
#     crop=0.66,
#     lr_flip=False,
#     tb_flip=False,
#     brightness=(0.5, 0.4, 4),
#     contrast=(0.5, 0.3, 5),
#     blur=1,
#     noise=10
#     )

# # Visualize samples of augmentations
# from viz import viz_sample_augmentations
# viz_sample_augmentations(data["X_train"], aug_func=aug_func, n_images=10, n_per_image=5, saveto=None)


# ##############################################################################
#                                                                           MAIN
# ##############################################################################
if __name__ == '__main__':
    # SETTINGS
    n_valid = 128
    data_file = "data_256.pickle"
    vgg16_snapshot = "/path/to/vgg16/vgg_16.ckpt"
    data = prepare_data(data_file, valid_from_train=True, n_valid=n_valid, max_data=None)
    n_classes = len(data["id2label"])

    print("DONE!!!")
