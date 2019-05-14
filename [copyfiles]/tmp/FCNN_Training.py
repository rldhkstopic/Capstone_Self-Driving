import numpy as np
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Activation, Dropout, UpSampling2D
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers


train_images = pickle.load(open("./pickles/train.p", "rb" ))
labels = pickle.load(open("./pickles/labels.p", "rb" ))

train_images = np.array(train_images)
labels = np.array(labels)
labels = labels / 255 # Normalize

train_images, labels = shuffle(train_images, labels)
X_train, X_val, y_train, y_val = train_test_split(train_images, labels, test_size=0.1)
# X_train = np.array(X_train)

batch_size = 128
epochs = 10
pool_size = (2, 2)
kernel_size = (3, 3)
input_shape = X_train.shape[1:]

model = Sequential()

model.add(BatchNormalization(input_shape=input_shape)) # Batch 정규화

model.add(Conv2D(8, kernel_size, padding='valid', strides=(1,1), activation = 'relu', name = 'Conv1'))
model.add(Conv2D(16, kernel_size, padding='valid', strides=(1,1), activation = 'relu', name = 'Conv2'))
model.add(MaxPooling2D(pool_size=pool_size))

model.add(Conv2D(16, kernel_size, padding='valid', strides=(1,1), activation = 'relu', name = 'Conv3'))
model.add(Dropout(0.2))
model.add(Conv2D(32, kernel_size, padding='valid', strides=(1,1), activation = 'relu', name = 'Conv4'))
model.add(Dropout(0.2))
model.add(Conv2D(32, kernel_size, padding='valid', strides=(1,1), activation = 'relu', name = 'Conv5'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=pool_size))

model.add(Conv2D(64, kernel_size, padding='valid', strides=(1,1), activation = 'relu', name = 'Conv6'))
model.add(Dropout(0.2))
model.add(Conv2D(64, kernel_size, padding='valid', strides=(1,1), activation = 'relu', name = 'Conv7'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(UpSampling2D(size=pool_size))

model.add(Conv2DTranspose(64, kernel_size, padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv1'))
model.add(Dropout(0.2))
model.add(Conv2DTranspose(64, kernel_size, padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv2'))
model.add(Dropout(0.2))
model.add(UpSampling2D(size=pool_size))

model.add(Conv2DTranspose(32, kernel_size, padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv3'))
model.add(Dropout(0.2))
model.add(Conv2DTranspose(32, kernel_size, padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv4'))
model.add(Dropout(0.2))
model.add(Conv2DTranspose(16, kernel_size, padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv5'))
model.add(Dropout(0.2))
model.add(UpSampling2D(size=pool_size))

model.add(Conv2DTranspose(16, kernel_size, padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv6'))
model.add(Conv2DTranspose(1, kernel_size, padding='valid', strides=(1,1), activation = 'relu', name = 'Final'))
"""
I had wanted to use dropout on every Convolutional and Deconvolutional layer,
but found it used up more memory than I had
"""
# conv2DTranspose : Conv를 거쳐 다운 샘플링된 맵을 사용하기 위한 업 샘플링 Conv 층

datagen = ImageDataGenerator(channel_shift_range=0.2)
datagen.fit(X_train)

# Compiling and training the model
model.compile(optimizer='Adam', loss='mean_squared_error')
model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                    steps_per_epoch=len(X_train)/batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_val, y_val)
                    )


model.trainable = True
model.compile(optimizer='Adam', loss='mean_squared_error')
model.save('LaneModel.h5')

model.summary()
