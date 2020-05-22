import numpy as np 
from sklearn.metrics import confusion_matrix, classification_report

from preprocessing_network import *

def print_predictions(my_model, gen, name):
    prediction = my_model.predict_generator(generator = gen)
    y_true = gen.classes
    y_pred = np.round(prediction)
    conf_mat = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    print(name + " set: Matrice de confusion et classification_report:")
    print(conf_mat)
    print(report)
