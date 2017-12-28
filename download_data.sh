LABEL_MAP_URL=http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/data/label_colors.txt
LABELS_URL=http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/data/LabeledApproved_full.zip
IMAGES_URL=http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/files/701_StillsRaw_full.zip

wget -c $LABEL_MAP_URL

wget -c $LABELS_URL
unzip -d labels LabeledApproved_full.zip

wget -c $IMAGES_URL
unzip -d inputs 701_StillsRaw_full.zip
