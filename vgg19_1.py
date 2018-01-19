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

# setup
os.chdir('/n/denic_lab/Users/nweir/statoil_odyssey/')

# load data
x_train = np.load('norm_im_arr.npy')
y_train = np.load('training_labs.npy')

def random_flips(im_arr, flip_frac):
    """Arbitrarily flip the above fraction of images in the array."""
    inds_to_flip = np.random.randint(0, im_arr.shape[0], int(im_arr.shape[0]*flip_frac))
    im_arr[inds_to_flip, :, :, :] = im_arr[inds_to_flip, ::-1, :, :]
    return im_arr

norm_w_flips = random_flips(x_train, 0.5)
norm_w_flips = np.append(norm_w_flips, np.zeros((1604,75,75,1)), axis=3)

vgg = applications.VGG19(weights = 'imagenet', include_top=False, input_shape=(75,75,3))

for layer in vgg.layers:
    layer.trainable = False

x = vgg.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
pred = Dense(1, activation='sigmoid')(x)

model_final = Model(input = vgg.input, output=pred)

model_final.compile(loss = 'binary_crossentropy', optimizer=optimizers.SGD(lr=0.001, momentum=0.9), metrics=['accuracy'])

checkpoint = ModelCheckpoint("vgg19_{epoch:02d}-{val_loss:.3f}.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')
model_final.summary()
model_final.fit(norm_w_flips, y_train, batch_size=32, epochs=100, verbose=2,
                                        validation_split = 0.2,
                                        callbacks = [checkpoint, early])
