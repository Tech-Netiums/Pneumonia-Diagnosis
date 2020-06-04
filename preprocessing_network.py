import numpy as np
from keras.preprocessing.image import ImageDataGenerator


#définition de la taille des images et de la taille des batch
BATCH_SIZE = 10
IMG_HEIGHT = 100
IMG_WIDTH = 100
NC = 3

#création d'un dataset avec ImageDataGenerator

#On normalise les pixels entre 0 et 1, et on centre 
image_generator = ImageDataGenerator(rescale=1./255, samplewise_center=True)
image_generator_augmented = ImageDataGenerator(rescale=1./255, samplewise_center=True, rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1)

#on crée les 3 dataset train/test/val
#on fixe les batch_size, les target_size, 2 classes, et on mélange
#les images sont déjà trié par classes dans les sous dossiers, on a juste à spécifier le class_mode = binary
#POUR L'ENTRAINEMENT
#train_gen_shuffle = image_generator.flow_from_directory('train/', class_mode='binary', target_size = (IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, shuffle = True)
train_gen_shuffle_augmented = image_generator_augmented.flow_from_directory('train/', class_mode='binary', target_size = (IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, shuffle = True)
#POUR EVALUER
train_gen = image_generator.flow_from_directory('train/', class_mode='binary', target_size = (IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, shuffle = False)
test_gen = image_generator.flow_from_directory('test/', class_mode='binary', target_size = (IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, shuffle = False)
#val_gen = image_generator.flow_from_directory('val/', class_mode='binary', target_size = (IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, shuffle = False)


#Remarque : train_gen est un DirectoryIterator : il contient une liste de (x,y), chaque (x,y) correspond à un batch
#(x,y) est composé de : x de taille batch_size*target_size*channels (3 channels car RGB) et y de taille batch_size




