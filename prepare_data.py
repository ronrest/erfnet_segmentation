from __future__ import print_function, division, unicode_literals
import glob
import os

colormap = {
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

# Check nothing stupid happened with mappings
assert set(colormap) == set(id2label) == set(label2id.keys()), "Something is wrong with the id label maps"


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


