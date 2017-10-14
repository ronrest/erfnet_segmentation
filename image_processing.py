import PIL
from PIL import ImageEnhance, Image, ImageFilter, ImageChops
import PIL.ImageOps
import numpy as np
randint = np.random.randint

__author__ = "Ronny Restrepo"
__copyright__ = "Copyright 2017, Ronny Restrepo"
__credits__ = ["Ronny Restrepo"]
__license__ = "Apache License"
__version__ = "2.0"


# ==============================================================================
#                                                           GET_ARRAY_COLOR_MODE
# ==============================================================================
def get_array_color_mode(x):
    """ Given a numpy array representing a single image, it returns the
        PIL color mode that will most likely work with it """
    x = x.squeeze()
    if x.ndim == 2:
        mode = "L"
    elif x.ndim == 3 and x.shape[2] == 1:
        mode = "L"
        x = x.squeeze()
    elif x.ndim == 3:
        mode = "RGB"
    else:
        assert False, "Incapable of interpreting array as an image"

    return mode


# ==============================================================================
#                                                                      PIL2ARRAY
# ==============================================================================
def pil2array(im):
    """ Given a PIL image it returns a numpy array representation """
    return np.asarray(im, dtype=np.uint8)


# ==============================================================================
#                                                                      ARRAY2PIL
# ==============================================================================
def array2pil(x):
    """ Given a numpy array containing image information returns a PIL image.
        Automatically handles mode, and even handles greyscale images with a
        channels axis
    """
    x = x.squeeze()
    return PIL.Image.fromarray(x, mode=get_array_color_mode(x))


# ==============================================================================
#                                                                  BATCH_PROCESS
# ==============================================================================
def batch_process(X, shape=None, mode=None):
    """ Given a batch of images as a numpy array, it does batch processing steps
        like resizing and color mode conversion.

        shape should be a 2-tuple (width, height) of the new desired
        dimensions.

        mode should be one of {"RGB", "L"}
    """
    assert X.dtype == np.uint8, "X should be 8 bit unsigned integers"
    if shape:
        width, height = shape
    else:
        width, height = X.shape[2], X.shape[1]

    n_samples = X.shape[0]

    # INITIALIZE NEW_BATCH ARRAY - By determining appropriate dimensions first
    if mode == "RGB" and X.ndim in {3,4}:
        n_channels = 3
        new_batch = np.zeros([n_samples, height, width, n_channels], dtype=np.uint8)
    elif mode == "L" and X.ndim in {3,4}:
        n_channels = 1
        new_batch = np.zeros([n_samples, height, width, n_channels], dtype=np.uint8)
    elif X.ndim == 4:
        n_channels = X.shape[3]
        new_batch = np.zeros([n_samples, height, width, n_channels], dtype=np.uint8)
    elif X.ndim == 3:
        n_channels = None
        new_batch = np.zeros([n_samples, height, width], dtype=np.uint8)
    else:
        assert False, "Cannot interpret X as a batch of images, check the dimensions"

    # PREPROCESS EACH IMAGE
    for i in range(n_samples):
        if shape:
            img = array2pil(X[i]).resize(shape, PIL.Image.BICUBIC)
        if mode in {"RGB", "L"}:
            img = img.convert(mode)

        if n_channels==1:
            img = np.asarray(img, dtype=np.uint8)
            new_batch[i] = np.expand_dims(img, axis=2)
        else:
            new_batch[i] = np.asarray(img, dtype=np.uint8)
    return new_batch


# ==============================================================================
#                                                                    RANDOM_CROP
# ==============================================================================
def random_crop(im, min_scale=0.5, max_scale=1.0, preserve_size=False, resample=PIL.Image.NEAREST):
    """
    Args:
        im:         PIL image
        min_scale:   (float) minimum ratio along each dimension to crop from.
        max_scale:   (float) maximum ratio along each dimension to crop from.
        preserve_size: (bool) Should it resize back to original dims?
        resample:       resampling method during rescale.

    Returns:
        PIL image of size crop_size, randomly cropped from `im`.
    """
    assert (min_scale < max_scale), "min_scale MUST be smaller than max_scale"
    width, height = im.size
    crop_width = np.random.randint(width*min_scale, width*max_scale)
    crop_height = np.random.randint(height*min_scale, height*max_scale)
    x_offset = np.random.randint(0, width - crop_width + 1)
    y_offset = np.random.randint(0, height - crop_height + 1)
    im2 = im.crop((x_offset, y_offset,
                   x_offset + crop_width,
                   y_offset + crop_height))
    if preserve_size:
        im2 = im2.resize(im.size, resample=resample)
    return im2


# ==============================================================================
#                                                         CROP_AND_PRESERVE_SIZE
# ==============================================================================
def crop_and_preserve_size(im, crop_dims, offset, resample=PIL.Image.NEAREST):
    """ Given a PIL image, the dimensions of the crop, and the offset of
        the crop, it crops the image, and resizes it back to the original
        dimensions.

    Args:
        im:         (PIL image)
        crop_dims:  Dimensions of the crop region [width, height]
        offset:     Position of the crop box from Top Left corner [x, y]
        resample:   resamplimg method
    """
    crop_width, crop_height = crop_dims
    x_offset, y_offset = offset
    im2 = im.crop((x_offset, y_offset,
                   x_offset + crop_width,
                   y_offset + crop_height))
    im2 = im2.resize(im.size, resample=resample)
    return im2


# ==============================================================================
#                                                             RANDOM_90_ROTATION
# ==============================================================================
def random_90_rotation(im):
    """ Randomly rotates image in 90 degree increments
        (90, -90, or 180 degrees) """
    methods = [PIL.Image.ROTATE_90, PIL.Image.ROTATE_180, PIL.Image.ROTATE_270]
    method = np.random.choice(methods)
    return im.transpose(method=method)


# ==============================================================================
#                                                                 RANDOM_LR_FLIP
# ==============================================================================
def random_lr_flip(im):
    """ Randomly flips the image left-right with 0.5 probablility """
    if np.random.choice([0,1]) == 1:
        return im.transpose(method=PIL.Image.FLIP_LEFT_RIGHT)
    else:
        return im


# ==============================================================================
#                                                                 RANDOM_TB_FLIP
# ==============================================================================
def random_tb_flip(im):
    """ Randomly flips the image top-bottom with 0.5 probablility """
    if np.random.choice([0,1]) == 1:
        return im.transpose(method=PIL.Image.FLIP_TOP_BOTTOM)
    else:
        return im


# ==============================================================================
#                                                                  RANDOM_INVERT
# ==============================================================================
def random_invert(im):
    """ With a 0.5 probability, it inverts the colors
        NOTE: This does not work on RGBA images yet. """
    assert im.mode != "RGBA", "Does random_invert not support RGBA images"
    if np.random.choice([0,1]) == 1:
        return PIL.ImageOps.invert(im)
    else:
        return im


# ==============================================================================
#                                                                   RANDOM_SHIFT
# ==============================================================================
def random_shift(im, max=(5,5)):
    """ Randomly shifts an image.

    Args:
        im: (pil image)
        max: (tuple of two ints) max amount in each x y direction.
    """
    x_offset = np.random.randint(0, max[0])
    y_offset = np.random.randint(0, max[1])
    return ImageChops.offset(im, xoffset=x_offset, yoffset=y_offset)


# ==============================================================================
#                                                                    SHIFT_IMAGE
# ==============================================================================
def shift_image(im, shift):
    """ Returns a shifted copy of a PIL image.
    Args:
        im:     (PIL image)
        shift:  (tuple of two ints) How much to shift along each axis (x, y)
    """
    return ImageChops.offset(im, xoffset=shift[0], yoffset=shift[1])


# ==============================================================================
#                                                              RANDOM_BRIGHTNESS
# ==============================================================================
def random_brightness(im, sd=0.5, min=0, max=20):
    """Creates a new image which randomly adjusts the brightness of `im` by
       randomly sampling a brightness value centered at 1, with a standard
       deviation of `sd` from a normal distribution. Clips values to a
       desired min and max range.

    Args:
        im:   PIL image
        sd:   (float) Standard deviation used for sampling brightness value.
        min:  (int or float) Clip contrast value to be no lower than this.
        max:  (int or float) Clip contrast value to be no higher than this.


    Returns:
        PIL image with brightness randomly adjusted.
    """
    brightness = np.clip(np.random.normal(loc=1, scale=sd), min, max)
    enhancer = ImageEnhance.Brightness(im)
    return enhancer.enhance(brightness)


# ==============================================================================
#                                                                RANDOM_CONTRAST
# ==============================================================================
def random_contrast(im, sd=0.5, min=0, max=10):
    """Creates a new image which randomly adjusts the contrast of `im` by
       randomly sampling a contrast value centered at 1, with a standard
       deviation of `sd` from a normal distribution. Clips values to a
       desired min and max range.

    Args:
        im:   PIL image
        sd:   (float) Standard deviation used for sampling contrast value.
        min:  (int or float) Clip contrast value to be no lower than this.
        max:  (int or float) Clip contrast value to be no higher than this.

    Returns:
        PIL image with contrast randomly adjusted.
    """
    contrast = np.clip(np.random.normal(loc=1, scale=sd), min, max)
    enhancer = ImageEnhance.Contrast(im)
    return enhancer.enhance(contrast)


# ==============================================================================
#                                                                    RANDOM_BLUR
# ==============================================================================
def random_blur(im, min=0, max=5):
    """ Creates a new image which applies a random amount of Gaussian Blur, with
        a blur radius that is randomly chosen to be in the range [min, max]
        inclusive.

    Args:
        im:   PIL image
        min:  (int) Min amount of blur desired.
        max:  (int) Max amount of blur desired.


    Returns:
        PIL image with random amount of blur applied.
    """
    blur_radius = randint(min, max+1)
    if blur_radius == 0:
        return im
    else:
        return im.filter(ImageFilter.GaussianBlur(radius=blur_radius))


# ==============================================================================
#                                                                   RANDOM_NOISE
# ==============================================================================
def random_noise(im, sd=5):
    """Creates a new image which has random noise.
       The intensity of the noise is determined by first randomly choosing the
       standard deviation of the noise as a value between 0 to `sd`.
       This value is then used as the standard deviation for randomly sampling
       individual pixel noise from a normal distribution.
       This random noise is added to the original image pixel values, and
       clipped to keep all values between 0-255.

    Args:
        im:   PIL image
        sd:   (int) Max Standard Deviation to select from.

    Returns:
        PIL image with random noise added.
    """
    mode = im.mode
    noise_sd = np.random.randint(0, sd)
    if noise_sd > 0:
        noise = np.random.normal(loc=0, scale=noise_sd, size=np.shape(im))
        im2 = np.asarray(im, dtype=np.float32) # prevent overflow
        im2 = np.clip(im2 + noise, 0, 255).astype(np.uint8)
        return array2pil(im2)
    else:
        return im


# ==============================================================================
#                                                                  RANDOM_SHADOW
# ==============================================================================
def random_shadow(im, shadow, intensity=(0.0, 0.7), crop_range=(0.02, 0.25)):
    """ Given an image of the scene, and an image of a shadow pattern,
        It will take random crops from the shadow pattern, and perform
        random rotations and flips of that crop, before overlaying the
        shadow on the scene image.

        The intensity of the shadow is randomly chosen from zero
        intensity to a max of `max_intensity`.

        NOTE: This was designed to make use of shadow images being
        black and white in color (but same colorspace mode as scene image).

    Args:
        im:             (PIL image) Image of scene
        shadow:         (PIL image)
                        Image of shadow pattern to take a crop from
        intensity:      (tuple of two floats)(default = (0.0, 0.7))
                        Min and max values (between 0 to 1) specifying how
                        strong to make the shadows.
        crop_range:     (tuple of two floats)(default=(0.02, 0.25))
                        Min and Max scale for random crop sizes from
                        the shadow image.
    Examples:
        shadow = PIL.Image.open("shadow_aug.png")
        image = PIL.Image.open("scene.jpg")
        random_shadow(image, shadow=shadow, max_intensity=0.7, crop_range=(0.02, 0.4))
    """
    width, height = im.size
    mode = im.mode
    assert im.mode == shadow.mode, "Scene image and shadow image must be same colorspace mode"

    # Take random crop from shadow image
    min_crop_scale, max_crop_scale = crop_range
    shadow = random_crop(shadow, min_scale=min_crop_scale, max_scale=max_crop_scale, preserve_size=False)
    shadow = shadow.resize((width, height), resample=PIL.Image.BILINEAR)

    # random flips, rotations, and color inversion
    shadow = random_tb_flip(random_lr_flip(random_90_rotation(shadow)))
    shadow = random_invert(shadow)
    # Ensure same shape as scene image after flips and rotations
    shadow = shadow.resize((width, height), resample=PIL.Image.BILINEAR)

    # Scale the shadow into proportional intensities (0-1)
    intensity_value = np.random.rand(1)
    min, max = intensity
    intensity_value = (intensity_value*(max - min))+min # remapped to min,max range
    shadow = np.divide(shadow, 255)
    shadow = np.multiply(intensity_value, shadow)

    # Overlay the shadow
    overlay = (np.multiply(im, 1-shadow)).astype(np.uint8)
    return PIL.Image.fromarray(overlay, mode=mode)


# ==============================================================================
#                                                                RANDOM_ROTATION
# ==============================================================================
def random_rotation(im, max=10, include_corners=True, resample=PIL.Image.NEAREST):
    """ Creates a new image which is rotated by a random amount between
        [-max, +max] inclusive.

    Args:
        im:              (PIL image)
        max:             (int) Max angle (in degrees in either direction).
        include_corners: (bool)
                If True, then the image canvas is expanded at first to
                fit the rotated corners, and then rescaled back to
                original image size.

                If False, then the original image canvas remains intact,
                and the corners of the rotated image that fall outside
                this box are clipped off.
    Returns:
        PIL image with random rotation applied.
    """
    original_dims = im.size
    angle = randint(-max, max+1)
    if angle == 0:
        return im
    else:
        im2 = im.rotate(angle, resample=resample, expand=include_corners)
        if include_corners:
            im2 = im2.resize(original_dims, resample=resample)
        return im2


# ==============================================================================
#                                                         RANDOM_TRANSFORMATIONS
# ==============================================================================
def random_transformations(
    X,
    shadow=(0.6, 0.9),
    shadow_file="shadow_pattern.jpg",
    shadow_crop_range=(0.02, 0.5),
    rotate=180,
    crop=0.5,
    lr_flip=True,
    tb_flip=True,
    brightness=(0.5, 0.4, 4),
    contrast=(0.5, 0.3, 5),
    blur=3,
    noise=10
    ):
    """ Takes a batch of input images `X` as a numpy array, and does random
        image transormations on them.

        NOTE:  Assumes the pixels for input images are in the range of 0-255.

    Args:
        X:                  (numpy array) batch of imput images
        Y:                  (numpy array) batch of segmentation labels
        shadow:             (tuple of two floats) (min, max) shadow intensity
        shadow_file:        (str) Path fo image file containing shadow pattern
        shadow_crop_range:  (tuple of two floats) min and max proportion of
                            shadow image to take crop from.
        shadow_crop_range:  ()(default=(0.02, 0.25))
        rotate:             (int)(default=180)
                            Max angle to rotate in each direction
        crop:               (float)(default=0.5)
        lr_flip:            (bool)(default=True)
        tb_flip:            (bool)(default=True)
        brightness:         ()(default=) (std, min, max)
        contrast:           ()(default=) (std, min, max)
        blur:               ()(default=3)
        noise:              ()(default=10)
    """
    # TODO: Random warping
    img_shape = X[0].shape
    images = np.zeros_like(X)
    n_images = len(images)

    if shadow is not None:
        assert shadow[0] < shadow[1], "shadow max should be greater than shadow min"
        shadow_image = PIL.Image.open(shadow_file)
        # Ensure shadow is same color mode as input images
        shadow_image = shadow_image.convert(get_array_color_mode(X[0]))

    for i in range(n_images):
        image = array2pil(X[i])
        original_dims = image.size

        if shadow is not None:
            image = random_shadow(image, shadow=shadow_image, intensity=shadow, crop_range=shadow_crop_range)

        if rotate:
            angle = randint(-rotate, rotate+1)
            image = image.rotate(angle, resample=PIL.Image.BICUBIC, expand=True)
            # No resizing is done yet to make the random crop high quality

        if crop is not None:
            # Crop dimensions
            min_scale = crop
            width, height = np.array(image.size)
            crop_width = np.random.randint(width*min_scale, width)
            crop_height = np.random.randint(height*min_scale, height)
            x_offset = np.random.randint(0, width - crop_width + 1)
            y_offset = np.random.randint(0, height - crop_height + 1)

            # Perform Crop
            image = image.crop((x_offset, y_offset, x_offset + crop_width, y_offset + crop_height))

        # Scale back after crop and rotate are done
        if rotate or crop:
            image = image.resize(original_dims, resample=PIL.Image.BICUBIC)

        if lr_flip and np.random.choice([True, False]):
            image = image.transpose(method=PIL.Image.FLIP_LEFT_RIGHT)

        if tb_flip and np.random.choice([True, False]):
            image=  image.transpose(method=PIL.Image.FLIP_TOP_BOTTOM)

        if brightness is not None:
            image = random_brightness(image, sd=brightness[0], min=brightness[1], max=brightness[2])
        if contrast is not None:
            image = random_contrast(image, sd=contrast[0], min=contrast[1], max=contrast[2])
        if blur is not None:
            image = random_blur(image, 0, blur)

        if noise:
            image = random_noise(image, sd=noise)

        # Put into array
        images[i] = np.asarray(image, dtype=np.uint8).reshape(img_shape)
    return images


# ==============================================================================
#                                                       CREATE_AUGMENTATION_FUNC
# ==============================================================================
def create_augmentation_func(**kwargs):
    """ Creates a function that performs random transformations on a
        X, Y pair of images for segmentation.

    Args:
        shadow:             (tuple of two floats) (min, max) shadow intensity
        shadow_file:        (str) Path fo image file containing shadow pattern
        shadow_crop_range:  (tuple of two floats) min and max proportion of
                            shadow image to take crop from.
        shadow_crop_range:  ()(default=(0.02, 0.25))
        rotate:             (int)(default=180)
                            Max angle to rotate in each direction
        crop:               (float)(default=0.5)
        lr_flip:            (bool)(default=True)
        tb_flip:            (bool)(default=True)
        brightness:         ()(default=) (std, min, max)
        contrast:           ()(default=) (std, min, max)
        blur:               ()(default=3)
        noise:              ()(default=10)

    Returns: (func)
        `augmentation_func` With the following API:
        `augmentation_func(X, Y)`

    Example:
        aug_func = create_augmentation_func(
            shadow=(0.01, 0.8),
            shadow_file="shadow_pattern.jpg",
            shadow_crop_range=(0.02, 0.5),
            rotate=30,
            crop=0.66,
            lr_flip=True,
            tb_flip=False,
            brightness=(0.5, 0.4, 4),
            contrast=(0.5, 0.3, 5),
            blur=2,
            noise=10
            )
        X_transformed = aug_func(X)
    """
    def augmentation_func(X):
        return random_transformations(X=X, **kwargs)
    return augmentation_func


# ==============================================================================
#                                        RANDOM_TRANSFORMATIONS_FOR_SEGMENTATION
# ==============================================================================
def random_transformations_for_segmentation(
    X,
    Y,
    shadow=(0.6, 0.9),
    shadow_file="shadow_pattern.jpg",
    shadow_crop_range=(0.02, 0.5),
    rotate=180,
    crop=0.5,
    lr_flip=True,
    tb_flip=True,
    brightness=(0.5, 0.4, 4),
    contrast=(0.5, 0.3, 5),
    blur=3,
    noise=10
    ):
    """ Takes a batch of input images `X`, segmentation labels `Y` as arrays,
        and does random image transormations on them.

        Ensures that any tansformations that shift or scale the input images
        also have the same transormations applied to the label images.

        NOTE:  Assumes the pixels for input images are in the range of 0-255.

    Args:
        X:                  (numpy array) batch of imput images
        Y:                  (numpy array) batch of segmentation labels
        shadow:             (tuple of two floats) (min, max) shadow intensity
        shadow_file:        (str) Path fo image file containing shadow pattern
        shadow_crop_range:  (tuple of two floats) min and max proportion of
                            shadow image to take crop from.
        shadow_crop_range:  ()(default=(0.02, 0.25))
        rotate:             (int)(default=180)
                            Max angle to rotate in each direction
        crop:               (float)(default=0.5)
        lr_flip:            (bool)(default=True)
        tb_flip:            (bool)(default=True)
        brightness:         ()(default=) (std, min, max)
        contrast:           ()(default=) (std, min, max)
        blur:               ()(default=3)
        noise:              ()(default=10)
    """
    # TODO: Random warping
    img_shape = X[0].shape
    label_shape = Y[0].shape
    images = np.zeros_like(X)
    labels = np.zeros_like(Y)
    n_images = len(images)

    if shadow is not None:
        assert shadow[0] < shadow[1], "shadow max should be greater than shadow min"
        shadow_image = PIL.Image.open(shadow_file)
        # Ensure shadow is same color mode as input images
        shadow_image.convert(get_array_color_mode(X[0]))

    for i in range(n_images):
        image = array2pil(X[i])
        label = array2pil(Y[i])
        original_dims = image.size

        if shadow is not None:
            image = random_shadow(image, shadow=shadow_image, intensity=shadow, crop_range=shadow_crop_range)

        if rotate:
            angle = randint(-rotate, rotate+1)
            image = image.rotate(angle, resample=PIL.Image.BICUBIC, expand=True)
            label = label.rotate(angle, resample=PIL.Image.NEAREST, expand=True)
            # No resizing is done yet to make the random crop high quality

        if crop is not None:
            # Crop dimensions
            min_scale = crop
            width, height = np.array(image.size)
            crop_width = np.random.randint(width*min_scale, width)
            crop_height = np.random.randint(height*min_scale, height)
            x_offset = np.random.randint(0, width - crop_width + 1)
            y_offset = np.random.randint(0, height - crop_height + 1)

            # Perform Crop
            image = image.crop((x_offset, y_offset, x_offset + crop_width, y_offset + crop_height))
            label = label.crop((x_offset, y_offset, x_offset + crop_width, y_offset + crop_height))

        # Scale back after crop and rotate are done
        if rotate or crop:
            image = image.resize(original_dims, resample=PIL.Image.BICUBIC)
            label = label.resize(original_dims, resample=PIL.Image.NEAREST)

        if lr_flip and np.random.choice([True, False]):
            image = image.transpose(method=PIL.Image.FLIP_LEFT_RIGHT)
            label = label.transpose(method=PIL.Image.FLIP_LEFT_RIGHT)

        if tb_flip and np.random.choice([True, False]):
            image=  image.transpose(method=PIL.Image.FLIP_TOP_BOTTOM)
            label=  label.transpose(method=PIL.Image.FLIP_TOP_BOTTOM)

        if brightness is not None:
            image = random_brightness(image, sd=brightness[0], min=brightness[1], max=brightness[2])
        if contrast is not None:
            image = random_contrast(image, sd=contrast[0], min=contrast[1], max=contrast[2])
        if blur is not None:
            image = random_blur(image, 0, blur)

        if noise:
            image = random_noise(image, sd=noise)

        # Put into array
        images[i] = np.asarray(image, dtype=np.uint8).reshape(img_shape)
        labels[i] = np.asarray(label, dtype=np.uint8).reshape(label_shape)
    return images, labels


# ==============================================================================
#                                      CREATE_AUGMENTATION_FUNC_FOR_SEGMENTATION
# ==============================================================================
def create_augmentation_func_for_segmentation(**kwargs):
    """ Creates a function that performs random transformations on a
        X, Y pair of images for segmentation.

    Args:
        shadow:             (tuple of two floats) (min, max) shadow intensity
        shadow_file:        (str) Path fo image file containing shadow pattern
        shadow_crop_range:  (tuple of two floats) min and max proportion of
                            shadow image to take crop from.
        shadow_crop_range:  ()(default=(0.02, 0.25))
        rotate:             (int)(default=180)
                            Max angle to rotate in each direction
        crop:               (float)(default=0.5)
        lr_flip:            (bool)(default=True)
        tb_flip:            (bool)(default=True)
        brightness:         ()(default=) (std, min, max)
        contrast:           ()(default=) (std, min, max)
        blur:               ()(default=3)
        noise:              ()(default=10)

    Returns: (func)
        `augmentation_func` With the following API:
        `augmentation_func(X, Y)`

    Example:
        aug_func = create_augmentation_func_for_segmentation(
            shadow=(0.01, 0.8),
            shadow_file="shadow_pattern.jpg",
            shadow_crop_range=(0.02, 0.5),
            rotate=30,
            crop=0.66,
            lr_flip=True,
            tb_flip=False,
            brightness=(0.5, 0.4, 4),
            contrast=(0.5, 0.3, 5),
            blur=2,
            noise=10
            )
        X_transformed, Y_transformed = aug_func(X, Y)
    """
    def augmentation_func(X, Y):
        return random_transformations_for_segmentation(X=X, Y=Y, **kwargs)
    return augmentation_func
