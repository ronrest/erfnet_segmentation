from __future__ import print_function, division, unicode_literals
import numpy as np
import tensorflow as tf

# from model_base import ImageClassificationModel
# from model_base import PretrainedImageClassificationModel
from data_processing import prepare_data, calculate_class_weights
from model_base import SegmentationModel

from erfnet import erfnetA

__author__ = "Ronny Restrepo"
__copyright__ = "Copyright 2017, Ronny Restrepo"
__credits__ = ["Ronny Restrepo"]
__license__ = "Apache License"
__version__ = "2.0"


# ##############################################################################
#                                                                   AUGMENTATION
# ##############################################################################
from image_processing import create_augmentation_func

aug_func = create_augmentation_func(
    shadow=(0.01, 0.8),
    shadow_file="shadow_pattern.jpg",
    shadow_crop_range=(0.02, 0.5),
    rotate=30,
    crop=0.66,
    lr_flip=False,
    tb_flip=False,
    brightness=(0.5, 0.4, 4),
    contrast=(0.5, 0.3, 5),
    blur=1,
    noise=10
    )

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

    # PREPARE DATA
    DATA_LIMIT = None
    data = prepare_data(data_file, valid_from_train=True, n_valid=n_valid, max_data=DATA_LIMIT)
    n_classes = len(data["id2label"])

    # CLASS WEIGHTS
    class_weights = calculate_class_weights(data["Y_train"], n_classes=n_classes, method="paszke", c=1.10)
    # class_weights = calculate_class_weights(data["Y_train"], n_classes=n_classes, method="logeigen2")


    model = SegmentationModel("erfnetA_01", img_shape=[256,256], n_classes=len(data["id2label"]))
    model.set_class_weights(class_weights)
    model.create_graph_from_logits_func(erfnetA)
    model.train(data, n_epochs=10, print_every=1, batch_size=8, alpha=5e-4, dropout=0.3, aug_func=None, viz_every=1)
    print("DONE!!!")
