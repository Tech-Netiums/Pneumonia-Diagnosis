import numpy as np
import keras
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from plot_utils import *

#PREPROCESSING

#définition de la taille des images et de la taille des batch
BATCH_SIZE = 10
IMG_HEIGHT = 100
IMG_WIDTH = 100

#création d'un dataset avec ImageDataGenerator

#On normalise les pixels entre 0 et 1
image_generator = ImageDataGenerator(rescale=1./255)

#on crée les 3 dataset train/test/val
#on fixe les batch_size, les target_size, 2 classes, et on mélange
#les images sont déjà trié par classes dans les sous dossiers, on a juste à spécifier le class_mode = binary
train_gen = image_generator.flow_from_directory('train/', class_mode='binary', target_size = (IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, shuffle = True)
test_gen = image_generator.flow_from_directory('test/', class_mode='binary', target_size = (IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, shuffle = True)
val_gen = image_generator.flow_from_directory('val/', class_mode='binary', target_size = (IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, shuffle = True)

#Remarque : train_gen est un DirectoryIterator : il contient une liste de (x,y), chaque (x,y) correspond à un batch
#(x,y) est composé de : x de taille batch_size*target_size*channels (3 channels car RGB) et y de taille batch_size

#print(train_gen.class_indices)
#{'NORMAL': 0, 'PNEUMONIA': 1}

#print(train_gen.image_shape)
#(300, 300, 3) #300 par 300, 3 layers RGB, tout est OK

#show(train_gen[0][0], n_cols=3) #impression d'un batch de x avec la fonction show() de plot_utils


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  #pour la binary classification

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
print(model.summary())
model.fit_generator(generator = train_gen, epochs=30)

#evaluation : il faut qu'on regarde plus de metriques : F1, accuracy, recall, VP/FP, VN/FN et matrices de confusion
score_train = model.evaluate_generator(generator = train_gen)
score_test = model.evaluate_generator(generator = test_gen)
score_val = model.evaluate_generator(generator = val_gen)
print(score_train)
print(score_test)
print(score_val)

