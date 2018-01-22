import pandas as pd
import os
import numpy as np
import scipy.ndimage as ndi
from scipy import stats
from skimage import io
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, ZeroPadding2D, AveragePooling2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.regularizers import l2
from keras import initializers
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from sklearn.model_selection import StratifiedKFold
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as k
from keras import optimizers
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau

# setup
os.chdir('/n/denic_lab/Users/nweir/statoil_odyssey/')

# load data
x_train = np.load('norm_im_arr.npy')
y_train = np.load('training_labs.npy')

def Model_3(conv_depth=32, lr = 0.001, use_dropout=True, dropout_p=0.2,
             conv_reg=0, dense_reg=0, adam_decay=0):
    #Building the model
    gmodel=Sequential()
    #Conv Layer 1
    gmodel.add(ZeroPadding2D(padding=(1, 1), input_shape=(75, 75, 3)))
    gmodel.add(Conv2D(conv_depth, kernel_size=(3, 3),activation='relu',
                     kernel_regularizer=l2(conv_reg)))
    gmodel.add(ZeroPadding2D(padding=(1, 1), input_shape=(75, 75, 2)))
    gmodel.add(Conv2D(conv_depth, kernel_size=(3, 3),activation='relu',
                     kernel_regularizer=l2(conv_reg)))
    gmodel.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    if use_dropout:
        gmodel.add(Dropout(dropout_p))

    #Conv Layer 2
    gmodel.add(ZeroPadding2D(padding=(1, 1)))
    gmodel.add(Conv2D(conv_depth*2, kernel_size=(3, 3),activation='relu',
                     kernel_regularizer=l2(conv_reg)))
    gmodel.add(ZeroPadding2D(padding=(1, 1)))
    gmodel.add(Conv2D(conv_depth*2, kernel_size=(3, 3),activation='relu',
                     kernel_regularizer=l2(conv_reg)))
    gmodel.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    if use_dropout:
        gmodel.add(Dropout(dropout_p))

    #Conv Layer 3
    gmodel.add(ZeroPadding2D(padding=(1, 1)))
    gmodel.add(Conv2D(conv_depth*4, kernel_size=(3, 3),activation='relu',
                     kernel_regularizer=l2(conv_reg)))
    gmodel.add(ZeroPadding2D(padding=(1, 1)))
    gmodel.add(Conv2D(conv_depth*4, kernel_size=(3, 3),activation='relu',
                     kernel_regularizer=l2(conv_reg)))
    gmodel.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    if use_dropout:
        gmodel.add(Dropout(dropout_p))

    #Conv Layer 4
    gmodel.add(ZeroPadding2D(padding=(1, 1)))
    gmodel.add(Conv2D(conv_depth*4, kernel_size=(3, 3),activation='relu',
                     kernel_regularizer=l2(conv_reg)))
    gmodel.add(ZeroPadding2D(padding=(1, 1)))
    gmodel.add(Conv2D(conv_depth*4, kernel_size=(3, 3),activation='relu',
                     kernel_regularizer=l2(conv_reg)))
    gmodel.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    if use_dropout:
        gmodel.add(Dropout(dropout_p))

    #Flatten the data for upcoming dense layers
    gmodel.add(Flatten())

    #Dense Layers
    gmodel.add(Dense(1024, activation='relu', kernel_regularizer=l2(dense_reg)))
    gmodel.add(Dropout(0.5))

    # Dense Layer 2
    gmodel.add(Dense(512, activation='relu', kernel_regularizer=l2(dense_reg)))
    gmodel.add(Dropout(0.5))

    # Sigmoid layer
    gmodel.add(Dense(1, activation='sigmoid'))

    mypotim=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=adam_decay)
    gmodel.compile(loss='binary_crossentropy',
                  optimizer=mypotim,
                  metrics=['accuracy'])
    gmodel.summary()
    return gmodel


current_gmodel=Model_3(conv_depth=32, lr=0.0001, use_dropout=True, dropout_p = 0.18)

batch_size = 32


norm_w_add = x_train
add_arr = norm_w_add[:,:,:,0] + norm_w_add[:,:,:,1]
add_arr = add_arr[:,:,:,np.newaxis]
norm_w_add = np.append(norm_w_add, add_arr, axis=3)
X_tr, X_val, y_tr, y_val = train_test_split(norm_w_add, y_train, test_size=.2, random_state=42)

train_datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True)

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(X_tr, y_tr,
        batch_size=batch_size)

validation_generator = test_datagen.flow(X_val, y_val,
        batch_size=batch_size)

current_gmodel.fit_generator(
    train_generator,
    steps_per_epoch=(norm_w_add.shape[0]/batch_size),
    epochs=200,
    verbose=2,
    validation_data=validation_generator,
    validation_steps=50,
    callbacks=[ModelCheckpoint(filepath="aug1-neg3lr-18dropout-32cd-{epoch:02d}-{val_loss:.3f}.hdf5",
                                     save_best_only=True),
                     EarlyStopping(patience=50),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, epsilon=1e-4, mode='min')])
