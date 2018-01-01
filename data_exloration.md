# Data
The Cambridge-driving Labeled Video Database (CamVid) is a dataset that contains 701 Images captured from the perspective of a car driving on the roads of Cambridge, UK. The images are labelled with 32 semantic classes that include things like the road, footpath, cars, pedestrians, trafic signs, etc.

## Input Data
The input images are RGB png images with dimensions of `960x720`. Below is a sample of 20 images from the dataset.


![sample of input images](sample_inputs.jpg)

## Labels
The labels are also encoded as RGB PNG images, with each of the 32 semantic classes represented as a different RGB value.

![Sample of label images](sample_labels.jpg)

The mapping of the different semantic classes is as follows:

<table border=0>
<tr>
    <td style="background-color:rgb(64, 128, 64);" width=100></td>
    <td>Animal (64, 128, 64)</td>
</tr>
<tr>
    <td style="background-color:rgb(192, 0, 128);" width=100></td>
    <td>Archway (192, 0, 128)</td>
</tr>
<tr>
    <td style="background-color:rgb(0, 128, 192);" width=100></td>
    <td>Bicyclist (0, 128, 192)</td>
</tr>
<tr>
    <td style="background-color:rgb(0, 128, 64);" width=100></td>
    <td>Bridge (0, 128, 64)</td>
</tr>
<tr>
    <td style="background-color:rgb(128, 0, 0);" width=100></td>
    <td>Building (128, 0, 0)</td>
</tr>
<tr>
    <td style="background-color:rgb(64, 0, 128);" width=100></td>
    <td>Car (64, 0, 128)</td>
</tr>
<tr>
    <td style="background-color:rgb(64, 0, 192);" width=100></td>
    <td>CartLuggagePram (64, 0, 192)</td>
</tr>
<tr>
    <td style="background-color:rgb(192, 128, 64);" width=100></td>
    <td>Child (192, 128, 64)</td>
</tr>
<tr>
    <td style="background-color:rgb(192, 192, 128);" width=100></td>
    <td>Column_Pole (192, 192, 128)</td>
</tr>
<tr>
    <td style="background-color:rgb(64, 64, 128);" width=100></td>
    <td>Fence (64, 64, 128)</td>
</tr>
<tr>
    <td style="background-color:rgb(128, 0, 192);" width=100></td>
    <td>LaneMkgsDriv (128, 0, 192)</td>
</tr>
<tr>
    <td style="background-color:rgb(192, 0, 64);" width=100></td>
    <td>LaneMkgsNonDriv (192, 0, 64)</td>
</tr>
<tr>
    <td style="background-color:rgb(128, 128, 64);" width=100></td>
    <td>Misc_Text (128, 128, 64)</td>
</tr>
<tr>
    <td style="background-color:rgb(192, 0, 192);" width=100></td>
    <td>MotorcycleScooter (192, 0, 192)</td>
</tr>
<tr>
    <td style="background-color:rgb(128, 64, 64);" width=100></td>
    <td>OtherMoving (128, 64, 64)</td>
</tr>
<tr>
    <td style="background-color:rgb(64, 192, 128);" width=100></td>
    <td>ParkingBlock (64, 192, 128)</td>
</tr>
<tr>
    <td style="background-color:rgb(64, 64, 0);" width=100></td>
    <td>Pedestrian (64, 64, 0)</td>
</tr>
<tr>
    <td style="background-color:rgb(128, 64, 128);" width=100></td>
    <td>Road (128, 64, 128)</td>
</tr>
<tr>
    <td style="background-color:rgb(128, 128, 192);" width=100></td>
    <td>RoadShoulder (128, 128, 192)</td>
</tr>
<tr>
    <td style="background-color:rgb(0, 0, 192);" width=100></td>
    <td>Sidewalk (0, 0, 192)</td>
</tr>
<tr>
    <td style="background-color:rgb(192, 128, 128);" width=100></td>
    <td>SignSymbol (192, 128, 128)</td>
</tr>
<tr>
    <td style="background-color:rgb(128, 128, 128);" width=100></td>
    <td>Sky (128, 128, 128)</td>
</tr>
<tr>
    <td style="background-color:rgb(64, 128, 192);" width=100></td>
    <td>SUVPickupTruck (64, 128, 192)</td>
</tr>
<tr>
    <td style="background-color:rgb(0, 0, 64);" width=100></td>
    <td>TrafficCone (0, 0, 64)</td>
</tr>
<tr>
    <td style="background-color:rgb(0, 64, 64);" width=100></td>
    <td>TrafficLight (0, 64, 64)</td>
</tr>
<tr>
    <td style="background-color:rgb(192, 64, 128);" width=100></td>
    <td>Train (192, 64, 128)</td>
</tr>
<tr>
    <td style="background-color:rgb(128, 128, 0);" width=100></td>
    <td>Tree (128, 128, 0)</td>
</tr>
<tr>
    <td style="background-color:rgb(192, 128, 192);" width=100></td>
    <td>Truck_Bus (192, 128, 192)</td>
</tr>
<tr>
    <td style="background-color:rgb(64, 0, 64);" width=100></td>
    <td>Tunnel (64, 0, 64)</td>
</tr>
<tr>
    <td style="background-color:rgb(192, 192, 0);" width=100></td>
    <td>VegetationMisc (192, 192, 0)</td>
</tr>
<tr>
    <td style="background-color:rgb(0, 0, 0);" width=100></td>
    <td>Void (0, 0, 0)</td>
</tr>
<tr>
    <td style="background-color:rgb(64, 192, 0);" width=100></td>
    <td>Wall (64, 192, 0)</td>
</tr>
</table>


## Distribution of classes

The distribution of how much of the image each class takes (as a  proportion of the number of pixels in the image) can be viewed in the following plot. We can see that buildings, roads, sky, and trees disproportionately dominate the scenes, and other classes occur with much less frequency.

![Image of violin plot](violin_plot.jpg)


## Data Preparation
### Resizing Input images
The input images were resized and reshaped to 256*256 in dimensions, and stored in numpy arrays of shape `[n_samples,256,256,3]`.

## Label images
The label images were also resized to 256*256 in dimensions. But in order to be useful for training a deep learning model, they had to be converted from RGB images to single channel images, with the iteger value for each pixel representing the class ID. These were also stored in numpy arrays, but of shape `[n_samples, 256,256]`.

### Train/validation split
Out of the 701 samples in the data, 128 were put aside at random to for the validation set. This left 573 images remaining for the training set.

The resulting data had the following shapes:

| Dataset | shape |
|---|---|
| `X_valid` | `[128, 256, 256, 3]` |
| `Y_valid` | `[128, 256, 256]` |
| `X_train` | `[573, 256, 256, 3]` |
| `Y_train` | `[573, 256, 256]` |

### Data Augmentation
Since the training set was quite small, data augmentation was needed for training in order to allow the model to generalize better.

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

### Class weighting
Since the distribution of classes on the images is quite imbalanced, class weighting was used to allow the model to learn about objects that it rarely sees, or which take up smaller regions of the total image. The method used was is from [Paszke et al 2016][paszke2016]

    weight_class = 1/ln(c + class_probability)

Where:

- `c` is some constant value that is manually set. As per [Romera et al 2017a][romera2017a], a value of `1.10` was used.

The table below shows the weight applied to the different classes when this formula is used. Greater weight is given to smaller or rarer objects, eg child (10.46), than objects that occur more often and consume large portions of the image, eg sky (4.37).

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
