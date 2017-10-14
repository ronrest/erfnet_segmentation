from __future__ import print_function, division, unicode_literals
import glob
import os
import PIL.Image
import numpy as np
import pickle

__author__ = "Ronny Restrepo"
__copyright__ = "Copyright 2017, Ronny Restrepo"
__credits__ = ["Ronny Restrepo"]
__license__ = "Apache License"
__version__ = "2.0"


label_colormap = {
    "Animal": (64, 128, 64),
    "Archway": (192, 0, 128),
    "Bicyclist": (0, 128, 192),
    "Bridge": (0, 128, 64),
    "Building": (128, 0, 0),
    "Car": (64, 0, 128),
    "CartLuggagePram": (64, 0, 192),
    "Child": (192, 128, 64),
    "Column_Pole": (192, 192, 128),
    "Fence": (64, 64, 128),
    "LaneMkgsDriv": (128, 0, 192),
    "LaneMkgsNonDriv": (192, 0, 64),
    "Misc_Text": (128, 128, 64),
    "MotorcycleScooter": (192, 0, 192),
    "OtherMoving": (128, 64, 64),
    "ParkingBlock": (64, 192, 128),
    "Pedestrian": (64, 64, 0),
    "Road": (128, 64, 128),
    "RoadShoulder": (128, 128, 192),
    "Sidewalk": (0, 0, 192),
    "SignSymbol": (192, 128, 128),
    "Sky": (128, 128, 128),
    "SUVPickupTruck": (64, 128, 192),
    "TrafficCone": (0, 0, 64),
    "TrafficLight": (0, 64, 64),
    "Train": (192, 64, 128),
    "Tree": (128, 128, 0),
    "Truck_Bus": (192, 128, 192),
    "Tunnel": (64, 0, 64),
    "VegetationMisc": (192, 192, 0),
    "Void": (0, 0, 0),
    "Wall": (64, 192, 0),
}


id2label = [
    'Void',
    'Sky',
    'Pedestrian',
    'Child',
    'Animal',
    'Tree',
    'VegetationMisc',
    'CartLuggagePram',
    'Bicyclist',
    'MotorcycleScooter',
    'Car',
    'SUVPickupTruck',
    'Truck_Bus',
    'Train',
    'OtherMoving',
    'Road',
    'RoadShoulder',
    'Sidewalk',
    'LaneMkgsDriv',
    'LaneMkgsNonDriv',
    'Bridge',
    'Tunnel',
    'Archway',
    'ParkingBlock',
    'TrafficLight',
    'SignSymbol',
    'Column_Pole',
    'Fence',
    'TrafficCone',
    'Misc_Text',
    'Wall',
    'Building',
]

label2id = {label:id for id, label in enumerate(id2label)}
idcolormap = [label_colormap[label] for label in id2label]

# Check nothing stupid happened with mappings
assert set(label_colormap) == set(id2label) == set(label2id.keys()), "Something is wrong with the id label maps"


# ==============================================================================
#                                                                 MAYBE_MAKE_DIR
# ==============================================================================
def maybe_make_dir(path):
    """ Checks if a directory path exists on the system, if it does not, then
        it creates that directory (and any parent directories needed to
        create that directory)
    """
    if not os.path.exists(path):
        os.makedirs(path)

# ==============================================================================
#                                                                     GET_PARDIR
# ==============================================================================
def get_pardir(file):
    """ Given a file path, it returns the parent directory of that file. """
    return os.path.dirname(file)


# ==============================================================================
#                                                              MAYBE_MAKE_PARDIR
# ==============================================================================
def maybe_make_pardir(file):
    """ Takes a path to a file, and creates the necessary directory structure
        on the system to ensure that the parent directory exists (if it does
        not already exist)
    """
    pardir = os.path.dirname(file)
    if pardir.strip() != "": # ensure pardir is not an empty string
        if not os.path.exists(pardir):
            os.makedirs(pardir)


# ==============================================================================
#                                                                       FILE2STR
# ==============================================================================
def file2str(file):
    """ Takes a file path and returns the contents of that file as a string."""
    with open(file, "r") as textFile:
        return textFile.read()

# ==============================================================================
#                                                                       STR2FILE
# ==============================================================================
def str2file(s, file, mode="w"):
    """ Writes a string to a file"""
    # Ensure parent directory and necesary file structure exists
    pardir = os.path.dirname(file)
    if pardir.strip() != "": # ensure pardir is not an empty string
        if not os.path.exists(pardir):
            os.makedirs(pardir)

    with open(file, mode=mode) as textFile:
        textFile.write(s)


# ==============================================================================
#                                                                     OBJ2PICKLE
# ==============================================================================
def obj2pickle(obj, file, protocol=2):
    """ Saves an object as a binary pickle file to the desired file path. """
    # Ensure parent directory and necesary file structure exists
    pardir = os.path.dirname(file)
    if pardir.strip() != "": # ensure pardir is not an empty string
        if not os.path.exists(pardir):
            os.makedirs(pardir)

    with open(file, mode="wb") as fileObj:
        pickle.dump(obj, fileObj, protocol=protocol)


# ==============================================================================
#                                                                     PICKLE2OBJ
# ==============================================================================
def pickle2obj(file):
    """ Loads the contents of a pickle as a python object. """
    with open(file, mode = "rb") as fileObj:
        obj = pickle.load(fileObj)
    return obj


# ==============================================================================
#                                                              CREATE_FILE_LISTS
# ==============================================================================
def create_file_lists(inputs_dir, labels_dir):
    """ Given the paths to the directories containing the input and label
        images, it creates a list of the full filepaths for those images,
        with the same ordering, so the same index in each list represents
        the corresponding input/label pair.

        Returns 2-tuple of two lists: (input_files, label_files)
    """
    # Create (synchronized) lists of full file paths to input and label images
    label_files = glob.glob(os.path.join(labels_dir, "*.png"))
    file_ids = [os.path.basename(f).replace("_L.png", ".png") for f in label_files]
    input_files = [os.path.join(inputs_dir, file_id) for file_id in file_ids]
    return input_files, label_files


# ==============================================================================
#                                                               CREATE_DATA_DICT
# ==============================================================================
def create_data_dict(datadir, X_train_subdir="train_inputs", Y_train_subdir="train_labels"):
    data = {}
    data["X_train"], data["Y_train"] = create_file_lists(
        inputs_dir=os.path.join(datadir, X_train_subdir),
        labels_dir=os.path.join(datadir, Y_train_subdir))
    return data


# ==============================================================================
#                                                              PIXELS_WITH_VALUE
# ==============================================================================
def pixels_with_value(img, val):
    """ Given an image as a numpy array, and a value representing the
        pixel values, eg [128,255,190] in an RGB image, then it returns
        a 2D boolean array with a True for every pixel position that has
        that value.
    """
    return np.all(img==np.array(val), axis=2)


# ==============================================================================
#                                                                   RGB2SEGLABEL
# ==============================================================================
def rgb2seglabel(img, colormap, channels_axis=False):
    """ Given an RGB image stored as a numpy array, and a colormap that
        maps from label id to an RGB color for that label, it returns a
        new numpy array with color chanel size of 1 where the pixel
        intensity values represent the class label id.

    Args:
        img:            (np array)
        colormap:       (list) list of pixel values for each class label id
        channels_axis:  (bool)(default=False) Should it return an array with a
                        third (color channels) axis of size 1?
    """
    height, width, _ = img.shape
    if channels_axis:
        label = np.zeros([height, width,1], dtype=np.uint8)
    else:
        label = np.zeros([height, width], dtype=np.uint8)
    for id in range(len(colormap)):
        label[np.all(img==np.array(idcolormap[id]), axis=2)] = id
    return label


# ==============================================================================
#                                                       LOAD_IMAGE_AND_SEGLABELS
# ==============================================================================
def load_image_and_seglabels(input_files, label_files, colormap, shape=(32,32), n_channels=3, label_chanel_axis=False):
    """ Given a list of input image file paths and corresponding segmentation
        label image files (with different RGB values representing different
        classes), and a colormap list, it:

        - loads up the images
        - resizes them to a desired shape
        - converts segmentation labels to single color channel image with
          integer value of pixel representing the class id.

    Args:
        input_files:        (list of str) file paths for input images
        label_files:        (list of str) file paths for label images
        colormap:           (list or None) A list where each index represents the
                            color value for the corresponding class id.
                            Eg: for RGB labels, to map class_0 to black and
                            class_1 to red:
                                [(0,0,0), (255,0,0)]
                            Set to None if images are already encoded as
                            greyscale where the integer value represents the
                            class id.
        shape:              (2-tuple of ints) (width,height) to reshape images
        n_channels:         (int) Number of chanels for input images
        label_chanel_axis:  (bool)(default=False) Use color chanel axis for
                            array of label images?
    """
    # Dummy proofing
    assert n_channels in {1,3}, "Incorrect value for n_channels. Must be 1 or 3. Got {}".format(n_channels)

    # Image dimensions
    width, height = shape
    n_samples = len(label_files)

    # Initialize input and label batch arrays
    X = np.zeros([n_samples, height, width, n_channels], dtype=np.uint8)
    if label_chanel_axis:
        Y = np.zeros([n_samples, height, width, 1], dtype=np.uint8)
    else:
        Y = np.zeros([n_samples, height, width], dtype=np.uint8)

    for i in range(n_samples):
        # Get filenames of input and label
        img_file = input_files[i]
        label_file = label_files[i]

        # Resize input and label images
        img = PIL.Image.open(img_file).resize(shape, resample=PIL.Image.CUBIC)
        label_img = PIL.Image.open(label_file).resize(shape, resample=PIL.Image.NEAREST)

        # Convert back to numpy arrays
        img = np.asarray(img, dtype=np.uint8)
        label_img = np.asarray(label_img, dtype=np.uint8)

        # Convert label image from RGB to single value int class labels
        if colormap is not None:
            label_img = rgb2seglabel(label_img, colormap=colormap, channels_axis=label_chanel_axis)

        # Add processed images to batch arrays
        X[i] = img
        Y[i] = label_img

    return X, Y


# ==============================================================================
#                                                                   PREPARE_DATA
# ==============================================================================
def prepare_data(data_file, valid_from_train=False, n_valid=1024, max_data=None, verbose=True):
    data = pickle2obj(data_file)

    # Create validation from train data
    if valid_from_train:
        data["X_valid"] = data["X_train"][:n_valid]
        data["Y_valid"] = data["Y_train"][:n_valid]
        data["X_train"] = data["X_train"][n_valid:]
        data["Y_train"] = data["Y_train"][n_valid:]

    if max_data:
        data["X_train"] = data["X_train"][:max_data]
        data["Y_train"] = data["Y_train"][:max_data]

    if verbose:
        # Print information about data
        print("DATA SHAPES")
        print("- X_valid: ", (data["X_valid"]).shape)
        print("- Y_valid: ", (data["Y_valid"]).shape)
        print("- X_train: ", (data["X_train"]).shape)
        print("- Y_train: ", (data["Y_train"]).shape)
        if "X_test" in data:
            print("- X_test: ", (data["X_test"]).shape)
            print("- Y_test: ", (data["Y_test"]).shape)

    return data


if __name__ == '__main__':
    # SETTINGS
    data_dir = "/path/to/camvid"
    data_dir = "/home/ronny/TEMP/camvid/"
    pickle_file = "data.pickle"
    shape = [128, 128]
    width, height = shape
    n_channels = 3
    label_chanel_axis=False # Create chanels axis for label images?

    print("CREATING DATA")
    print("- Getting list of files")
    file_data = create_data_dict(data_dir, X_train_subdir="train_inputs", Y_train_subdir="train_labels")
    n_samples = len(file_data["X_train"])

    est_size = n_samples*width*height*(3+1)/(1024*1000)
    print("- Estimated data size is {} MB (+ overhead)".format(est_size))

    print("- Loading image files and converting to arrays")
    data = {}
    data["X_train"], data["Y_train"] = load_image_and_seglabels(
        input_files=file_data["X_train"],
        label_files=file_data["Y_train"],
        colormap=idcolormap,
        shape=shape,
        n_channels=n_channels,
        label_chanel_axis=label_chanel_axis)

    # # Visualize the data
    # from viz import viz_overlayed_segmentation_label
    # gx = batch2grid(X, 5,5)
    # gy = batch2grid(Y, 5,5)
    # g = viz_overlayed_segmentation_label(gx, gy, colormap=idcolormap, alpha=0.7, saveto=None)
    # g.show()

    print("- Pickling the data to:", pickle_file)
    obj2pickle(data, pickle_file)
    print("- DONE!")
