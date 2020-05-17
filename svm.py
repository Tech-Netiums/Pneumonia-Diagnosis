# -*- coding: utf-8 -*-
"""
Created on Sun May 17 17:54:11 2020

@author: louis
"""
#%%
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from keras.preprocessing.image import ImageDataGenerator
from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split

from skimage.io import imread
from skimage.transform import resize
from pathlib import Path

#%%
def load_image_files(container_path, dimension=(64, 64, 3)):
    """
    Load image files with categories as subfolder names 
    which performs like scikit-learn sample dataset
    
    Parameters
    ----------
    container_path : string or unicode
        Path to the main folder holding one subfolder per category
    dimension : tuple
        size to which image are adjusted to
        
    Returns
    -------
    Bunch
    """
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    descr = "A image classification dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = imread(file)
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
            flat_data.append(img_resized.flatten()) 
            images.append(img_resized)
            target.append(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)

    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr)


#%%

train_set = load_image_files("C:/Users/louis/Documents/GitHub/Pneumonia-Diagnosis/train")
val_set = load_image_files("C:/Users/louis/Documents/GitHub/Pneumonia-Diagnosis/val")
test_set = load_image_files("C:/Users/louis/Documents/GitHub/Pneumonia-Diagnosis/test")

#%%
param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]
svc = svm.SVC()
clf = GridSearchCV(svc, param_grid)
clf.fit(train_set.data, train_set.target)

#%%
results = pd.DataFrame(clf.cv_results_)

trained_grid_search = pickle.dumps(clf)

#%%

pred = clf.predict(test_set.data)

#%%

print(metrics.accuracy_score(test_set.target, pred))
print(metrics.f1_score(test_set.target, pred))
print(metrics.confusion_matrix(test_set.target, pred))