

# coding: utf-8

# # Mask R-CNN - Inspect Training Data
# 
# Inspect and visualize data loading and pre-processing code.

# In[2]:

import os
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon

import utils
import visualize
from visualize import display_images
import model as modellib
from model import log



ROOT_DIR = os.getcwd()


# ## Configurations
# 
# Run one of the code blocks below to import and load the configurations to use.

# In[1]:

# Run one of the code blocks

# Shapes toy dataset
# import shapes
# config = shapes.ShapesConfig()

# MS COCO Dataset
np.random.seed(123)
import coco
config = coco.CocoConfig()
COCO_DIR = r'D:\data\coco'  # TODO: enter value here


# ## Dataset

# In[3]:

# Load dataset
if config.NAME == 'shapes':
    dataset = shapes.ShapesDataset()
    dataset.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
elif config.NAME == "coco":
    dataset = coco.CocoDataset()
    dataset.load_coco(COCO_DIR, "train")

# Must call before using the dataset
dataset.prepare()

print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
#for i, info in enumerate(dataset.class_info):
#    print("{:3}. {:50}".format(i, info['name']))


# ## Display Samples
# 
# Load and display images and masks.

# In[4]:

# Load and display random samples
image_id = np.random.choice(dataset.image_ids, 10)[5]
#for image_id in image_ids:
image = dataset.load_image(image_id)
mask, class_ids = dataset.load_mask(image_id)
visualize.display_top_masks(image, mask, class_ids, dataset.class_names)


