from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('AGG') # make matplotlib work on aws
import matplotlib.pyplot as plt
import seaborn as sns

import os
import PIL
import PIL.Image
import PIL.ImageChops


__author__ = "Ronny Restrepo"
__copyright__ = "Copyright 2017, Ronny Restrepo"
__credits__ = ["Ronny Restrepo"]
__license__ = "Apache License"
__version__ = "2.0"


# ==============================================================================
#                                                                   TRAIN_CURVES
# ==============================================================================
def train_curves(train, valid, saveto=None, title="Accuracy over time", ylab="accuracy", legend_pos="lower right"):
    """ Plots the training curves. If `saveto` is specified, it saves the
        the plot image to a file instead of showing it.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.suptitle(title, fontsize=15)
    ax.plot(train, color="#FF4F40",  label="train")
    ax.plot(valid, color="#307EC7",  label="valid")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylab)

    # Grid lines
    ax.grid(True)
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='#888888', linestyle='-')
    plt.grid(b=True, which='minor', color='#AAAAAA', linestyle='-', alpha=0.2)

    # Legend
    ax.legend(loc=legend_pos, title="", frameon=False,  fontsize=8)

    # Save or show
    if saveto is None:
        plt.show()
    else:
        fig.savefig(saveto)
        plt.close()



# ==============================================================================
#                                                                   BATCH 2 GRID
# ==============================================================================
def batch2grid(imgs, rows, cols):
    """
    Given a batch of images stored as a numpy array of shape:

           [n_batch, img_height, img_width]
        or [n_batch, img_height, img_width, n_channels]

    it creates a grid of those images of shape described in `rows` and `cols`.

    Args:
        imgs: (numpy array)
            Shape should be either:
                - [n_batch, im_rows, im_cols]
                - [n_batch, im_rows, im_cols, n_channels]

        rows: (int) How many rows of images to use
        cols: (int) How many cols of images to use

    Returns: (numpy array)
        The grid of images as one large image of either shape:
            - [n_classes*im_cols, num_per_class*im_rows]
            - [n_classes*im_cols, num_per_class*im_rows, n_channels]
    """
    # TODO: have a resize option to rescale the individual sample images
    # TODO: Have a random shuffle option
    # TODO: Set the random seed if needed
    assert rows>0 and cols>0, "rows and cols must be positive integers"

    # Prepare dimensions of the grid
    n_cells = (rows*cols)
    imgs = imgs[:n_cells] # Only use the number of images needed to fill grid
    n_samples, img_height, img_width, n_channels = imgs.shape

    # Image dimensions
    n_dims = imgs.ndim
    assert n_dims==3 or n_dims==4, "Incorrect # of dimensions for input array"

    # Deal with images that have no color channel
    if n_dims == 3:
        imgs = np.expand_dims(imgs, axis=3)

    # Handle case where there is not enough images in batch to fill grid
    if n_cells > n_samples:
        n_gap = n_cells - n_samples
        imgs = np.pad(imgs, pad_width=[(0,n_gap),(0,0), (0,0), (0,0)], mode="constant", constant_values=0)

    # Reshape into grid
    grid = imgs.reshape(rows, cols,img_height,img_width,n_channels).swapaxes(1,2)
    grid = grid.reshape(rows*img_height,cols*img_width,n_channels)

    # If input was flat images with no color channels, then flatten the output
    if n_dims == 3:
        grid = grid.squeeze(axis=2) # axis 2 because batch dim has been removed

    return grid



# ==============================================================================
#                                                   PLOT_SEG_LABEL_DISTRIBUTIONS
# ==============================================================================
def plot_seg_label_distributions(Y, id2label, colormap, saveto=None):
    """ Given an array of the segmentation labels in a dataset, it plots
        the relative distribution of each class label as violin plots.

        It shows the distribution of how much each class takes up as a
        proportion of the entire image (how many pixels are taken up by
        that class).

    Dependencies:
        seaborn, matplotlib, numpy, collections Counter

    Args:
        Y:          (np array) Array contianing the segmentation label images,
                    with integer vals representing the class ids for each pixel
        id2label:   (list) Map from label id to human readable string label
        colormap:   (list) List of the (R,G,B) values for each class.
        saveto:     (str or None)(default=None) optionally save the image
                    instead of displaying it.
    """
    rgb2hex = lambda x: "#"+"".join(["{:>02s}".format(hex(ch)[2:]) for ch in x])

    distributions = []
    n_classes = len(id2label)
    for img in Y:
        tally = Counter(img.flatten()) # counts for each class
        distributions.append([tally[i]/float(img.size) for i in range(n_classes)])
    distributions = np.array(distributions)

    sns.set(style="whitegrid", palette="pastel", color_codes=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 20))
    sns.violinplot(data=distributions,
                   scale="count",
                   ax=ax,
                   dodge=False,
                   fliersize=2,
                   linewidth=0.5,
                   inner="point",  #:“box”, “quartile”, “point”, “stick”, None
                   orient="h",
                   palette=[rgb2hex(code) for code in colormap],
                   )
    # fig.suptitle('Distribution of Space Taken up by Classes', fontsize=15)
    ax.set_title("Distribution of Space Taken up by Classes", fontdict={"weight": "bold", "size": 20})
    ax.set_xlabel("Proportion of Image", fontsize=20)
    ax.set_yticklabels(id2label)
    plt.setp(ax.get_xticklabels(), fontsize=20)
    plt.setp(ax.get_yticklabels(), fontsize=20)
    ax.set_xlim([0,0.7])
    fig.tight_layout()

    if saveto:
        plt.savefig(saveto)
        plt.close()
    else:
        plt.show()


# ==============================================================================
#                                                                      ARRAY2PIL
# ==============================================================================
def array2pil(x):
    """ Given a numpy array containing image information returns a PIL image.
        Automatically handles mode, and even handles greyscale images with a
        channels axis
    """
    if x.ndim == 2:
        mode = "L"
    elif x.ndim == 3 and x.shape[2] == 1:
        mode = "L"
        x = x.squeeze()
    elif x.ndim == 3:
        mode = "RGB"
    return PIL.Image.fromarray(x, mode=mode)


# ==============================================================================
#                                                       VIZ_SAMPLE_AUGMENTATIONS
# ==============================================================================
def viz_sample_segmentation_augmentations(X, Y, aug_func, colormap, n_images=5, n_per_image=5, saveto=None):
    """ Given a batch of data X, and Y,  it takes n_images samples, and performs
        `n_per_image` random transformations for segmentation data on each of
        those images. It then puts them in a grid to visualize. Grid size is:
            n_images wide x n_per_image high

    Args:
        X:          (np array) batch of images
        Y:          (np array) batch of labels
        aug_func:   (func) function with API `aug_func(X, Y)` that performs
                    random transformations on the images for segmentation
                    purposes.
        colormap:   (list of 3-tuples of ints)
                    A list where each index represents the RGB value
                    for the corresponding class id.
                    Eg: to map class_0 to black and class_1 to red:
                        [(0,0,0), (255,0,0)]

        n_images:   (int) Number of base images to use for visualization
        n_per_image:(int) Number of augmentations to show per base image
        saveto:     (str or None)

    Returns: (None, or PIL image)
        If saveto is provided, then it saves teh image and returns None.
        Else, it returns the PIL image.
    Examples:
        samples = viz_sample_seg_augmentations(data["X_train"], data["Y_train"],
            aug_func=aug_func, n_images=5, n_per_image=5, saveto=None)
        samples.show()
    """
    X = X[:n_images]
    Y = Y[:n_images]
    grid = []
    colormap = np.array(colormap).astype(np.uint8)

    # Perform Augmentations
    for col in range(n_per_image):
        x,y = aug_func(X, Y)
        y = colormap[y]
        z = np.append(x, y, axis=2) # append the label to the right
        grid.append(z)

    # Put into a grid
    _, height, width, n_channels = X.shape
    grid = np.array(grid, dtype=np.uint8).reshape(n_images*n_per_image, height, 2*width, n_channels)
    grid = batch2grid(grid, n_per_image, n_images)

    # Convert to PIL image
    grid = array2pil(grid)

    # Optionally save image
    if saveto is not None:
        # Create necessary file structure
        pardir = os.path.dirname(saveto)
        if pardir.strip() != "": # ensure pardir is not an empty string
            if not os.path.exists(pardir):
                os.makedirs(pardir)
        grid.save(saveto, "JPEG")

    return grid


# ==============================================================================
#                                                         VIZ_SEGMENTATION_LABEL
# ==============================================================================
def viz_segmentation_label(label, colormap=None, saveto=None):
    """ Given a 2D numpy array representing a segmentation label, with
        the pixel value representing the class of the object, then
        it creates an RGB PIL image that color codes each label.

    Args:
        label:      (numpy array) 2D flat image where the pixel value
                    represents the class label.
        colormap:   (list of 3-tuples of ints)
                    A list where each index represents the RGB value
                    for the corresponding class id.
                    Eg: to map class_0 to black and class_1 to red:
                        [(0,0,0), (255,0,0)]
                    By default, it creates a map that supports 4 classes:

                        0. black
                        1. guava red
                        2. nice green
                        3. nice blue

        saveto:         (str or None)(default=None)(Optional)
                        File path to save the image to (as a jpg image)
    Returns:
        PIL image
    """
    # Default colormap
    if colormap is None:
        colormap = [[0,0,0], [255,79,64], [115,173,33],[48,126,199]]

    # Map each pixel label to a color
    label_viz = np.zeros((label.shape[0],label.shape[1],3), dtype=np.uint8)
    uids = np.unique(label)
    for uid in uids:
        label_viz[label==uid] = colormap[uid]

    # Convert to PIL image
    label_viz = PIL.Image.fromarray(label_viz)

    # Optionally save image
    if saveto is not None:
        # Create necessary file structure
        pardir = os.path.dirname(saveto)
        if pardir.strip() != "": # ensure pardir is not an empty string
            if not os.path.exists(pardir):
                os.makedirs(pardir)
        label_viz.save(saveto, "JPEG")

    return label_viz


# ==============================================================================
#                                                                      ARRAY2PIL
# ==============================================================================
def array2pil(x):
    """ Given a numpy array containing image information returns a PIL image.
        Automatically handles mode, and even handles greyscale images with a
        channels axis
    """
    if x.ndim == 2:
        mode = "L"
    elif x.ndim == 3 and x.shape[2] == 1:
        mode = "L"
        x = x.squeeze()
    elif x.ndim == 3:
        mode = "RGB"
    return PIL.Image.fromarray(x, mode=mode)


# ==============================================================================
#                                               VIZ_OVERLAYED_SEGMENTATION_LABEL
# ==============================================================================
def viz_overlayed_segmentation_label(img, label, colormap=None, alpha=0.5, saveto=None):
    """ Given a base image, and the segmentation label image as numpy arrays,
        It overlays the segmentation labels on top of the base image, color
        coded for each separate class.

    Args:
        img:        (np array) numpy array containing base image (uint8 0-255)
        label:      (np array) numpy array containing segmentation labels,
                    with each pixel value representing the class label.
        colormap:   (None or list of 3-tuples) For each class label, specify
                    the RGB values to color code those pixels. Eg: red would
                    be `(255,0,0)`.
                    If `None`, then it supports up to 4 classes in a default
                    colormap:

                        0 = black
                        1 = red
                        2 = green
                        3 = blue

        alpha:      (float) Alpha value for overlayed segmentation labels
        saveto:     (None or str) Optional filepath to save this
                    visualization as a jpeg image.
    Returns:
        (PIL Image) PIL image of the visualization.
    """
    # Load the image
    img = array2pil(img)
    img = img.convert("RGB")

    # Default colormap
    if colormap is None:
        colormap = [[127,127,127],[255,0,0],[0,255,0],[0,0,255]]
    label = viz_segmentation_label(label, colormap=colormap)

    # Overlay the input image with the label
    overlay = PIL.ImageChops.blend(img, label, alpha=alpha)
    # overlay = PIL.ImageChops.add(img, label, scale=1.0)
    # overlay = PIL.ImageChops.screen(img, label)

    # Optionally save image
    if saveto is not None:
        # Create necessary file structure
        pardir = os.path.dirname(saveto)
        if pardir.strip() != "": # ensure pardir is not an empty string
            if not os.path.exists(pardir):
                os.makedirs(pardir)
        overlay.save(saveto, "JPEG")

    return overlay


# ==============================================================================
#                                                   VIZ_SAMPLE_SEG_AUGMENTATIONS
# ==============================================================================
def viz_sample_seg_augmentations(X, Y, aug_func, n_images=5, n_per_image=5, saveto=None):
    """ Given a batch of data X, and Y,  it takes n_images samples, and performs
        `n_per_image` random transformations for segmentation data on each of
        those images. It then puts them in a grid to visualize. Grid size is:
            n_images wide x n_per_image high

    Args:
        X:          (np array) batch of images
        Y:          (np array) batch of labels images
        aug_func:   (func) function with API `aug_func(X, Y)` that performs
                    random transformations on the images for segmentation
                    purposes.
        n_images:   (int)
        n_per_image:(int)
        saveto:     (str or None)

    Returns: (None, or PIL image)
        If saveto is provided, then it saves teh image and returns None.
        Else, it returns the PIL image.
    Examples:
        samples = viz_sample_seg_augmentations(data["X_train"], data["Y_train"],
            aug_func=aug_func, n_images=5, n_per_image=5, saveto=None)
        samples.show()
    """
    X = X[:n_images]
    Y = Y[:n_images]
    gx = []
    gy = []

    # Perform Augmentations
    for col in range(n_per_image):
        x, y = aug_func(X, Y)
        gx.append(x)
        gy.append(y)

    # Put into a grid
    _, height, width, n_channels = X.shape
    gx = np.array(gx, dtype=np.uint8).reshape(n_images*n_per_image, height, width, n_channels)
    gy = np.array(gy, dtype=np.uint8).reshape(n_images*n_per_image, height, width)
    gx = batch2grid(gx, n_per_image, n_images)
    gy = batch2grid(gy, n_per_image, n_images)

    # Overlay labels on top of image
    return viz_overlayed_segmentation_label(img=gx, label=gy, saveto=saveto)


# ==============================================================================
#                                                         VIZ_SEGMENTATION_PAIRS
# ==============================================================================
def viz_segmentation_pairs(X, Y, Y2=None, colormap=None, gridshape=(2,8), saveto=None):
    """ Given a batch of input images, and corresponding labels (and optionaly)
        a second set of labes (eg, predictions), it creates a grid, of
        image and label pairs/triplets, such that a [2,4] grid would look like:
            [ x ][ x ][ x ][ x ]
            [ y ][ y ][ y ][ y ]
            [ y2][ y2][ y2][ y2]
            [ x ][ x ][ x ][ x ]
            [ y ][ y ][ y ][ y ]
            [ y2][ y2][ y2][ y2]
    Args:
        X:          (numpy array) Batch of input images
        Y:          (numpy array) Batch of corresponding labels, of shape
                                  [n_batch, img_height, img_width]
                                  Each pixel value should be a class label for
                                  that pixel.
        Y2:         (numpy array) An optional second label, eg for predictions.
        colormap:   (numpy array) Each element contains the RGB 3-tuple that
                                  the corresponding class id maps to.
                                  eg: [(0,0,0), (255,0,0), (0,0,255)]
        gridshape:  (2-tuple)     (rows, cols)
        saveto:     (str or None) Where to save the visualization as an image.         
    """
    assert (X.ndim == 3) or (X.ndim == 4 and X.shape[-1] in {1,3}), "X is wrong dimensions"
    assert (Y.ndim == 3), "Y is wrong dimensions"
    assert (Y2 is None) or (Y2.ndim == 3), "Y2 is wrong dimensions"

    # LIMIT SAMPLES- Only use the number of images needed to fill grid
    rows, cols = gridshape
    assert rows>0 and cols>0, "rows and cols must be positive integers"
    n_cells = (rows*cols)
    X = X[:n_cells]
    n_samples = X.shape[0]

    # RESHAPE INPUT IMAGES - to include a color channels axis
    if (X.ndim == 3):
        X = np.expand_dims(X, axis=3)

    # SET COLORMAP
    if colormap is None:
        colormap = [[0,0,0], [255,79,64], [115,173,33],[48,126,199]]

    # ---------------------------------------
    # GROUP THE IMAGES - into pairs/triplets
    # ---------------------------------------
    output = []
    for i in range(min(n_cells, n_samples)):
        x = X[i]
        y = Y[i]

        # Convert greyscale images to RGB.
        if x.shape[-1] == 1:
            x = np.repeat(x, 3, axis=2)

        # Apply colormap to Y and Y2
        y = np.array(colormap)[y].astype(np.uint8)
        if Y2 is None:
            output.append(np.concatenate([x,y], axis=0))
        else:
            y2 = Y2[i]
            y2 = np.array(colormap)[y2].astype(np.uint8)
            output.append(np.concatenate([x,y,y2], axis=0))

    output = np.array(output, dtype=np.uint8)

    # ---------------------
    # CREATE GRID
    # ---------------------
    output = batch2grid(output, rows=rows, cols=cols)
    output = PIL.Image.fromarray(output.squeeze())

    # Optionally save image
    if saveto is not None:
        # Create necessary file structure
        pardir = os.path.dirname(saveto)
        if pardir.strip() != "": # ensure pardir is not an empty string
            if not os.path.exists(pardir):
                os.makedirs(pardir)
        output.save(saveto, "JPEG")

    return output


def vizseg(img, label, pred, saveto, colormap=None, gridshape=(2,8)):
    viz_segmentation_pairs(img, Y=label, Y2=pred, colormap=colormap, gridshape=gridshape, saveto=saveto)
