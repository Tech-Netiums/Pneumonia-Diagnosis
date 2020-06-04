import numpy as np
from keras.models import load_model

from model_network import *
from preprocessing_network import *
from scoring_network import *

#création du modèle
classifier = XRAY_classifier()
classifier.build_model(drop_rate=0.20, n_blocks=4)
#compilation du modèle
classifier.compile_model()
print(classifier.model.summary())
#entrainement du modèle 
classifier.fit_model(gen = train_gen_shuffle_augmented, n_epochs = 100, val=test_gen)
#saving
#classifier.model.save('2blocks_drop20_20epochs')
#appel
#model_loaded = load_model('2blocks_drop20_20epochs')
#evaluation 
print_predictions(classifier.model, train_gen, "Train")
#print_predictions(classifier.model, val_gen, "Validation")
print_predictions(classifier.model, test_gen, "Test")

