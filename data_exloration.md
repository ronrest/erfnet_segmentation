# Exploring the Camvid dataset

```py

```


# Data Exploration

The dataset contains 701 samples. The input images are RGB png images with dimensions of `960x720`. The label images are also encoded in RGB with the following RGB color mappings:

```py
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
```

The distribution of how much of the image each class takes (as a  proportion of the number of pixels in the image) can be viewed in the following plot. We can see that buildings, roads, sky, and trees disproportionately dominate the scenes, and other classes occur with much less frequency.

![Image of violin plot](violin_plot.jpg)


## Sample Data
Some sample input images.

![sample of input images](sample_inputs.jpg)

Label Images

![Sample of label images](sample_labels.jpg)


Labels overlayed on top of input images with opacity of 0.7.

![Sample images of training data and labels](sample_overlayed.jpg)



# Data Preparation
## Train/validation split
Out of the 701 samples in the data, 128 were put aside at random to for the validation set. This left 573 images remaining for the training set. Since the training set is quite small, data augmentation was needed for training to allow the model to generalize better.

## Data Augmentation
The following data augmentation steps were taken:

- **shadow augmentation**: with random blotches and shapes of different intensities being overlayed over the image.
- **random rotation**: between -30 to 30 degrees
- **random crops**: between 0.66 to 1.0 of the image
- **random brightness**: between 0.5 and 4, with a standard deviation of 0.5
- **random contrast**: between 0.3 and 5, with a standard deviation of 0.5
- **random blur**: randomly apply small amount of blur
- **random noise**: randomly shift the pixel intensities with a standard devition of 10.

Below is an example of two training images (and their accompanying labels) with data augmentation applied to them five times.

![Image of data augmentations](sample_augmentation_pairs.jpg)

## Class weighting
Since the distribution of classes on the images is quite imbalanced, class weighting was used to allow the model to learn about objects that it rarely sees, or which take up smaller regions of the total image. The method used was is from [Paszke et al 2016][paszke2016]

    weight_class = 1/ln(c + class_probability)

Where:

- `c` is some constant value that is manually set. As per TODO: XXX REFERENCE a value of `1.10` was used.


|Class|Weight|
|---|---|
| Void | 8.19641542163
| Sky | 4.3720296426
| Pedestrian | 9.88371158915
| Child | 10.4631076496
| Animal | 10.4870856376
| Tree | 5.43594061016
| VegetationMisc | 9.77766487673
| CartLuggagePram | 10.4613353846
| Bicyclist | 9.99897358787
| MotorcycleScooter | 10.4839211216
| Car | 7.95097080927
| SUVPickupTruck | 9.7926997517
| Truck_Bus | 10.0194993159
| Train | 10.4920586873
| OtherMoving | 10.1258833847
| Road | 3.15728309302
| RoadShoulder | 10.2342992955
| Sidewalk | 6.58893008318
| LaneMkgsDriv | 9.08285926082
| LaneMkgsNonDriv | 10.4757462996
| Bridge | 10.4371404226
| Tunnel | 10.4920560223
| Archway | 10.4356039685
| ParkingBlock | 10.1577327692
| TrafficLight | 10.1479245421
| SignSymbol | 10.3749043447
| Column_Pole | 9.60606490919
| Fence | 9.26646394904
| TrafficCone | 10.4882678542
| Misc_Text | 9.94147719336
| Wall | 9.32269173889
| Building | 3.47796865386


[paszke2016]: https://arxiv.org/abs/1606.02147
