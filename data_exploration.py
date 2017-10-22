import matplotlib.pyplot as plt
from viz import viz_overlayed_segmentation_label, viz_segmentation_label, batch2grid
from data_processing import pickle2obj
from data_processing import idcolormap, id2label

# ==============================================================================
#                                                                        RGB2HEX
# ==============================================================================
def rgb2hex(code):
    """ Given an iterable of integers representing an RGB or RGBA color
        it returns the hexadecimal version of that color:
    """
    hx = "#" + "".join(["{:>02s}".format(hex(channel)[2:])  for channel in code])
    return hx.upper()


# Get the data
pickle_file = "data.pickle"
data = pickle2obj(pickle_file)

# VISUALIZE THE DATA
gx = batch2grid(data["X_train"][:20], rows=5, cols=4)
PIL.Image.fromarray(gx).save("sample_inputs.jpg", "JPEG")

gy = batch2grid(data["Y_train"][:20], rows=5, cols=4)
viz_segmentation_label(gy, colormap=idcolormap, saveto="sample_labels.jpg")

g = viz_overlayed_segmentation_label(gx, gy, colormap=idcolormap, alpha=0.7, saveto="sample_overlayed.jpg")
g.show()


################################################################################
# CLASS DISTRIBUTION VIOLIN PLOTS
from viz import plot_seg_label_distributions
plot_seg_label_distributions(data["Y_train"], id2label=id2label, colormap=idcolormap, saveto="violin_plot.jpg")
