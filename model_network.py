import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.activations import relu, sigmoid
from keras.losses import binary_crossentropy
from keras.backend import mean as keras_mean

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
        self.batchn1 = BatchNormalization()(self.conv2)
        self.pool1 = MaxPooling2D((2, 2))(self.batchn1)
        self.drop1 = Dropout(drop_rate)(self.pool1)
        #block2
        self.conv3 = Conv2D(64, (3,3), activation='relu')(self.drop1)
        self.conv4 = Conv2D(64, (3,3), activation='relu')(self.conv3)
        self.batchn2 = BatchNormalization()(self.conv4)
        self.pool2 = MaxPooling2D((2, 2))(self.batchn2)
        self.drop2 = Dropout(drop_rate)(self.pool2)
        #block3
        if n_blocks >= 3:
            self.conv5 = Conv2D(64, (3,3), activation='relu')(self.drop2)
            self.conv6 = Conv2D(64, (3,3), activation='relu')(self.conv5)
            self.batchn3 = BatchNormalization()(self.conv6)
            self.pool3 = MaxPooling2D((2, 2))(self.batchn3)
            self.drop3 = Dropout(drop_rate)(self.pool3)
        #block4
        if n_blocks == 4:
            self.conv7 = Conv2D(64, (3,3), activation='relu')(self.drop3)
            self.conv8 = Conv2D(64, (3,3), activation='relu')(self.conv7)
            self.batchn4 = BatchNormalization()(self.conv8)
            self.pool4 = MaxPooling2D((2, 2))(self.batchn4)
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

    def custom_loss(y_true, y_pred):
        weights = 2*(1 - y_true) + 1.
        bce = binary_crossentropy(y_true, y_pred)
        weighted_bce = keras_mean(bce * weights)
        return weighted_bce

    import keras.losses
    keras.losses.custom_loss = custom_loss

    def compile_model(self):
        self.model.compile(loss='custom_loss', optimizer='rmsprop',metrics=['accuracy'])
    
    def fit_model(self, gen, n_epochs, val):
        training = self.model.fit_generator(generator = gen, epochs=n_epochs, validation_data=val)
        #print training
        print(training.history.keys())
        plt.plot(training.history['accuracy'])
        plt.plot(training.history['val_accuracy'])
        plt.title('model accuracy - 4 blocks + data augmentation - drop rate = 0.2')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

