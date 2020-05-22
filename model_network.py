import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from keras.activations import relu, sigmoid, tanh

IMG_HEIGHT = 100
IMG_WIDTH = 100
NC = 3

class XRAY_classifier:

    def __init__(self):
        self.input_shape = (IMG_HEIGHT,IMG_WIDTH,NC)

    def build_model(self, drop_rate, n_blocks=2):
        #modèle de type VGG avec des blocks CONV/RELU - CONV/RELU - MAXPOOLING - DROPOUT
        #2(par défaut),3 ou 4 blocks possibles, réglage du drop_rate possible
        #input
        self.inputs = Input(shape=self.input_shape)
        #block1
        self.conv1 = Conv2D(32, (3,3), activation='relu')(self.inputs)
        self.conv2 = Conv2D(32, (3,3), activation='relu')(self.conv1)
        self.pool1 = MaxPooling2D((2, 2))(self.conv2)
        self.drop1 = Dropout(drop_rate)(self.pool1)
        #block2
        self.conv3 = Conv2D(64, (3,3), activation='relu')(self.drop1)
        self.conv4 = Conv2D(64, (3,3), activation='relu')(self.conv3)
        self.pool2 = MaxPooling2D((2, 2))(self.conv4)
        self.drop2 = Dropout(drop_rate)(self.pool2)
        #block3
        if n_blocks >= 3:
            self.conv5 = Conv2D(64, (3,3), activation='relu')(self.drop2)
            self.conv6 = Conv2D(64, (3,3), activation='relu')(self.conv5)
            self.pool3 = MaxPooling2D((2, 2))(self.conv6)
            self.drop3 = Dropout(drop_rate)(self.pool3)
        #block4
        if n_blocks == 4:
            self.conv7 = Conv2D(64, (3,3), activation='relu')(self.drop3)
            self.conv8 = Conv2D(64, (3,3), activation='relu')(self.conv7)
            self.pool4 = MaxPooling2D((2, 2))(self.conv8)
            self.drop4 = Dropout(drop_rate)(self.pool4)
        #flatten
        if n_blocks == 2:
            self.flatt = Flatten()(self.drop2)
        if n_blocks == 3:
            self.flatt = Flatten()(self.drop3)
        if n_blocks == 4:
            self.flatt = Flatten()(self.drop4)
        #output
        self.dense1 = Dense(256, activation='relu')(self.flatt)
        self.drop5 =  Dropout(drop_rate)(self.dense1)
        self.dense2 = Dense(1, activation='sigmoid')(self.drop5)
        #construction du modèle
        self.model = Model(inputs=self.inputs, outputs=self.dense2)
        
    def compile_model(self):
        self.model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    
    def fit_model(self, gen, n_epochs, test):
        training = self.model.fit_generator(generator = gen, epochs=n_epochs, validation_data=test)
        #print training
        print(training.history.keys())
        plt.plot(training.history['accuracy'])
        plt.plot(training.history['val_accuracy'])
        plt.title('model accuracy - 2 blocks - drop rate = 0.2')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

