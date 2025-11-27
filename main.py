import os

from skimage.io import imread
from skimage.transform import resize
import numpy as np

# prepare data
input_dir = 'D:\\xiaoyangliu\\image_classification\\clf-data'
categories = ['empty', 'not_empty']

data = []
labels = []

# for all categories and category indices (as label)
for category_idx, category in enumerate(categories):
    # for all file under each category
    for file in os.listdir(os.path.join(input_dir, category)):
        # read out file path
        img_path = os.path.join(input_dir, category, file)
        # read out file
        img = imread(img_path)
        # resize image to 15x15
        img = resize(img, (15, 15))
        # convert the image into an array of 15x15x3 numbers
        #   where [B_1, G_1, R_1, B_2, G_2, R_2, ...]
        #   [B_1, G_1, R_1] is the first pixel
        data.append(img.flatten())
        labels.append(category_idx)

# save all numerical arrays in data
data = np.asarray(data)  
# save all labels in labels
labels = np.asarray(labels)

# train / test split

# train classifier

# test performance
