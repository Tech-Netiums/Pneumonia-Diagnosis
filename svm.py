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
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedShuffleSplit
from joblib import dump, load
from preprocessing import convolution2D, laplacian_of_gaussian_33, median_cut
from sklearn import preprocessing
from skimage.io import imread
from skimage.transform import resize
from pathlib import Path
from skimage.color import rgb2gray

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

train_set = load_image_files("C:/Users/louis/Documents/Louis/ecole/2A/Méthodes d'apprentissage/chest_xray/train")
test_set = load_image_files("C:/Users/louis/Documents/Louis/ecole/2A/Méthodes d'apprentissage/chest_xray/test")


#%%
scaler = preprocessing.StandardScaler().fit(train_set.data)
normed_train_set = scaler.transform(train_set.data,True)
normed_test_set = scaler.transform(test_set.data, True)

#%%

gray_train_set = [rgb2gray(img) for img in train_set.images]
gray_test_set =  [rgb2gray(img) for img in test_set.images]

#%%

laplacian_train_set = [convolution2D(img, laplacian_of_gaussian_33).flatten() for img in gray_train_set]
laplacian_test_set = [convolution2D(img, laplacian_of_gaussian_33).flatten() for img in gray_test_set]

#%%

median_cut_train_set = [median_cut(img).flatten() for img in gray_train_set] 
median_cut_test_set = [median_cut(img).flatten() for img in gray_test_set]

#%%
param_grid = [
  {'C': [1, 10, 100, 1000], 'gamma': [0.0001,0.001, 0.01], 'kernel': ['rbf']},
 ]
svc = svm.SVC()
clf = GridSearchCV(svc, param_grid)
clf.fit(median_cut_train_set, train_set.target)

#%%
dump(clf, 'median_cut_grid_search.joblib') 
results = pd.DataFrame(clf.cv_results_)

#%%

clf = load('trained_grid_search.joblib') 

results = pd.DataFrame(clf.cv_results_)

#%%

pred = clf.predict(median_cut_test_set)


#%%

print(metrics.accuracy_score(test_set.target, pred))
print(metrics.f1_score(test_set.target, pred))
print(metrics.confusion_matrix(test_set.target, pred))


#%%

param_grid2 = [{'C': [100, 1000, 10000], 'gamma': [0.01,0.001], 'kernel': ['rbf']}]
clf2 = GridSearchCV(svc, param_grid2)
clf2.fit(train_set.data, train_set.target)
#%%
dump(clf2, 'trained_grid_search2.joblib') 

#%%

results2 = pd.DataFrame(clf2.cv_results_)

#%%
pred = clf2.predict(test_set.data)
print(metrics.accuracy_score(test_set.target, pred))
print(metrics.f1_score(test_set.target, pred))
print(metrics.confusion_matrix(test_set.target, pred))


#%% 
param_grid3 = [{'C': [100, 1000], 'gamma': [0.01,0.001], 'kernel': ['rbf'], 'class_weight' : ['balanced'] }]
clf3 = GridSearchCV(svc, param_grid3)
clf3.fit(train_set.data, train_set.target)

#%%

dump(clf3, 'trained_grid_search3.joblib')

#%% 

results3 = pd.DataFrame(clf3.cv_results_)
pred = clf3.predict(test_set.data)
print(metrics.accuracy_score(test_set.target, pred))
print(metrics.f1_score(test_set.target, pred))
print(metrics.confusion_matrix(test_set.target, pred))

#%%

param_grid4 = [{'C': [100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}]
clf4 = GridSearchCV(svc, param_grid4)
clf4.fit(train_set.data, train_set.target)
#%%
dump(clf4, 'trained_grid_search4.joblib')

#%%

clf6 = svm.SVC(C=100000000, gamma= 0.1).fit(median_cut_train_set, train_set.target)


#%%

pred = clf6.predict(median_cut_test_set)
print(metrics.accuracy_score(test_set.target, pred))
print(metrics.f1_score(test_set.target, pred))
print(metrics.confusion_matrix(test_set.target, pred))

#%%

C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid5 = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
clf5 = GridSearchCV(svc, param_grid5, cv = cv)
clf5.fit(train_set.data, train_set.target)



clf5 = svm.SVC(C= 10000, gamma= 0.0001, kernel = 'rbf')