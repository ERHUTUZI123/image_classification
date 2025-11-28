import os
import pickle

from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# prepare data
input_dir = 'D:\\xiaoyangliu\\image_classification\\clf-data'
categories = ['empty', 'not_empty']

# create default data and labels
data = []
labels = []

# loop over all categories and category indices (as label)
for category_idx, category in enumerate(categories):
    # loop over all file under each category
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
# test_size=20%: 20% data for test and 80% data for train
# shuffle=True: shuffle data to avoid bias 
# stratify=labels:
#   Perform a stratified split. The train and test sets will preserve 
#   the same class proportion as in the original dataset.
#   Example: if the original labels are 80% red and 20% blue,
#   then both the train set and test set will maintain 80% red
#   and 20% blue.
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# train classifier
classifier = SVC()

parameters = [{'gamma':[0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

grid_search = GridSearchCV(classifier, parameters)


grid_search.fit(x_train, y_train)

# test performance
best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(x_test)

score = accuracy_score(y_prediction, y_test)

print('{}% of samples were correctly classified'.format(str(score * 100)))

pickle.dump(best_estimator, open('./model.p', 'wb'))
