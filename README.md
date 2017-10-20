# Semantic Segmentation on CamVid Dataset



## Get the data

- [camvid dataset page](http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/)

Each of the files can be downloaded using:

```sh
LABEL_MAP_URL=http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/data/label_colors.txt
LABELS_URL=http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/data/LabeledApproved_full.zip
IMAGES_URL=http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/files/701_StillsRaw_full.zip

wget -c $LABEL_MAP_URL

wget -c $LABELS_URL
unzip -d train_labels LabeledApproved_full.zip

wget -c $IMAGES_URL
unzip 701_StillsRaw_full.zip
mv 701_StillsRaw_full train_inputs
```
